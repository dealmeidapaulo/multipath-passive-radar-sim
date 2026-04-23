from __future__ import annotations
from typing import List
import numpy as np
import concurrent.futures
import os
import itertools
from .ray import Ray
from .propagation import compute_fspl, sample_reflection_attenuation, compute_sphere_rcs_bounce_gain, compute_scattered_doppler

def _aabb_hit(origin, direction, bmin, bmax, t_min=1e-5):
    t_enter, t_exit = -np.inf, np.inf
    normal_enter    = np.zeros(3)
    for ax in range(3):
        d = direction[ax]
        if abs(d) < 1e-12:
            if origin[ax] < bmin[ax] or origin[ax] > bmax[ax]: return None, None
            continue
        t1 = (bmin[ax] - origin[ax]) / d
        t2 = (bmax[ax] - origin[ax]) / d
        if t1 > t2: t1, t2 = t2, t1
        if t1 > t_enter:
            t_enter = t1
            n = np.zeros(3); n[ax] = -np.sign(d)
            normal_enter = n
        t_exit = min(t_exit, t2)
    if t_enter > t_exit or t_exit < t_min: return None, None
    if t_enter < t_min: return t_exit, -normal_enter
    return t_enter, normal_enter

def _floor_hit(origin, direction, box):
    if direction[2] > -1e-10: return None, None
    t = (box.box_min[2] - origin[2]) / direction[2]
    if t < 1e-5: return None, None
    hit = origin + t * direction
    if (box.box_min[0] <= hit[0] <= box.box_max[0] and box.box_min[1] <= hit[1] <= box.box_max[1]):
        return t, np.array([0.0, 0.0, 1.0])
    return None, None

def _t_exit_domain(origin, direction, box):
    t_best = None
    for ax, val in [(0, box.box_min[0]), (0, box.box_max[0]), (1, box.box_min[1]), (1, box.box_max[1]), (2, box.box_max[2])]:
        d = direction[ax]
        if abs(d) < 1e-12: continue
        t = (val - origin[ax]) / d
        if t > 1e-5 and (t_best is None or t < t_best): t_best = t
    return t_best

def _sphere_hit(origin, direction, center, radius, t_min=1e-5):
    oc = origin - center
    a  = float(np.dot(direction, direction))
    b  = 2.0 * float(np.dot(oc, direction))
    c  = float(np.dot(oc, oc)) - radius**2
    disc = b*b - 4*a*c
    if disc < 0: return None
    sq = np.sqrt(disc)
    for t in [(-b - sq)/(2*a), (-b + sq)/(2*a)]:
        if t > t_min: return t
    return None

def _nearest_hit(origin, direction, scene):
    t_exit = _t_exit_domain(origin, direction, scene.box)
    best_t, best_n, best_kind = None, None, ''

    def _upd(t, n, kind):
        nonlocal best_t, best_n, best_kind
        if t is not None and (best_t is None or t < best_t):
            best_t, best_n, best_kind = t, n, kind

    t, n = _floor_hit(origin, direction, scene.box)
    _upd(t, n, 'floor')
    
    for obs in scene.obstacles:
        t, n = _aabb_hit(origin, direction, obs.box_min, obs.box_max)
        _upd(t, n, 'obstacle')
        
    if getattr(scene, 'uav', None) is not None:
        t_uav = _sphere_hit(origin, direction, scene.uav.position, scene.uav.radius)
        if t_uav is not None:
            hit_pt = origin + t_uav * direction
            n_uav = (hit_pt - scene.uav.position) / scene.uav.radius
            _upd(t_uav, n_uav, 'uav')

    if t_exit is not None and (best_t is None or t_exit <= best_t):
        return t_exit, None, 'exit'
    return best_t, best_n, best_kind


def scatter_ray(d_in: np.ndarray, normal: np.ndarray, roughness: float) -> np.ndarray:
    """
    Blends perfect specular reflection with a cosine-weighted Lambertian lobe.
    roughness: 0.0 (pure mirror) to 1.0 (pure diffuse scattering)
    """
    # 1. Pure specular reflection
    specular = d_in - 2.0 * float(np.dot(d_in, normal)) * normal
    if roughness < 1e-5:
        return specular

    # 2. Cosine-weighted hemisphere sample
    r1, r2 = np.random.rand(), np.random.rand()
    phi = 2.0 * np.pi * r1
    sin_theta = np.sqrt(r2)
    cos_theta = np.sqrt(1.0 - r2)

    # Local Cartesian coordinates
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = cos_theta  # Z aligns with the normal

    # 3. Build Orthonormal Basis (ONB) around the normal
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    tangent = np.cross(helper, normal)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)

    # Transform diffuse sample to world space
    diffuse = x * tangent + y * bitangent + z * normal

    # 4. Blend specular and diffuse based on roughness
    d_out = (1.0 - roughness) * specular + roughness * diffuse
    d_out /= np.linalg.norm(d_out)
    
    # Failsafe: if the blend pushes the vector inside the wall, force pure diffuse
    if np.dot(d_out, normal) <= 0:
        return diffuse
        
    return d_out


def ray_is_occluded_by_uav(ray: Ray, uav) -> bool:
    if uav is None:
        return False
    for i in range(len(ray.points) - 1):
        A = ray.points[i]
        B = ray.points[i+1]
        AB = B - A
        seg_len = float(np.linalg.norm(AB))
        if seg_len < 1e-9: continue
        dir_AB = AB / seg_len
        t = _sphere_hit(A, dir_AB, uav.position, uav.radius)
        if t is not None and t < seg_len:
            return True
    return False

def _trace_chunk(args):
    dirs_chunk, scene, tx_obj, debug_tx, rx_pos, rx_rad = args
    results = []
    fc = tx_obj.frequency
    initial_pwr = tx_obj.tx_power_dbm

    for d_init in dirs_chunk:
        pos = tx_obj.position.copy()
        d = d_init / np.linalg.norm(d_init)
        points = [pos.copy()]
        current_power = initial_pwr
        current_doppler = 0.0
        hit_uav_flag = False

        for bounce in range(scene.n_max):
            t_rx = _sphere_hit(pos, d, rx_pos, rx_rad)
            t_wall, normal, kind = _nearest_hit(pos, d, scene)

            if t_rx is not None and (t_wall is None or t_rx < t_wall):
                hit_pt = pos + t_rx * d
                points.append(hit_pt.copy())
                
                if getattr(scene, 'use_physics', False):
                    current_power -= compute_fspl(t_rx, fc)
                    if current_power <= getattr(scene, 'noise_floor_dbm', -110.0): break
                
                r = Ray(tx_obj.tx_id, points, d.copy(), fc, current_power)
                r.is_uav_bounce = hit_uav_flag
                r.doppler_shift = current_doppler
                results.append(r)
                break

            if kind == 'exit' or t_wall is None:
                if debug_tx: 
                    dist = t_wall if t_wall is not None else 150.0
                    points.append(pos + dist * d)
                    results.append(Ray(tx_obj.tx_id, points, d.copy(), fc, current_power))
                break
            
            hit_pt = pos + t_wall * d
            points.append(hit_pt.copy())
            
            if getattr(scene, 'use_physics', False):
                current_power -= compute_fspl(t_wall, fc)
                if kind == 'uav':
                    current_power += compute_sphere_rcs_bounce_gain(scene.uav.radius, fc)
                else:
                    current_power += sample_reflection_attenuation() 
                    
                if current_power <= getattr(scene, 'noise_floor_dbm', -110.0): 
                    if debug_tx: results.append(Ray(tx_obj.tx_id, points, d.copy(), fc, current_power))
                    break
            elif bounce == scene.n_max - 1:
                if debug_tx: results.append(Ray(tx_obj.tx_id, points, d.copy(), fc, current_power))
                break

            if kind == 'uav':
                hit_uav_flag = True
                v_out = d - 2.0 * np.dot(d, normal) * normal
                v_out /= np.linalg.norm(v_out)
                current_doppler += compute_scattered_doppler(scene.uav.velocity, d, v_out, fc)
                d = v_out
            else:
#                d = d - 2.0 * np.dot(d, normal) * normal
#                d /= np.linalg.norm(d)
                # Fetch roughness from scene, default to 0.0 (mirror) if not set
                roughness = getattr(scene, 'roughness', 0.0)
                d = scatter_ray(d, normal, roughness)

            
            pos = hit_pt + 1e-5 * d

    return results

def trace(scene, receiver=None, dedup_tol: float = 0.5, debug_tx: bool = False) -> List[Ray]:
    golden  = (1.0 + np.sqrt(5.0)) / 2.0
    indices = np.arange(scene.n_rays)
    theta   = np.arccos(1.0 - 2.0*(indices+0.5)/scene.n_rays)
    phi     = 2.0*np.pi*indices/golden
    dirs    = np.column_stack([
        np.sin(theta)*np.cos(phi),
        np.sin(theta)*np.sin(phi),
        np.cos(theta),
    ])

    all_results = []
    max_workers = os.cpu_count() or 4
    
    for tx_obj in scene.transmitters:
        chunk_size = int(np.ceil(len(dirs) / max_workers))
        dir_chunks = [dirs[i:i + chunk_size] for i in range(0, len(dirs), chunk_size)]
        _rx_pos = receiver.position if receiver is not None else scene.box.box_max / 2
        _rx_rad = receiver.radius if receiver is not None else 2.0
        worker_args = [(chunk, scene, tx_obj, debug_tx, _rx_pos, _rx_rad) for chunk in dir_chunks]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            chunk_results = list(executor.map(_trace_chunk, worker_args))
            
        all_results.extend(itertools.chain.from_iterable(chunk_results))

    dec = max(0, int(-np.log10(dedup_tol)))
    seen = set()
    unique = []
    for r in all_results:
        # FIX: Deduplicate based on the exact spatial hit point on the RX sphere,
        # ensuring multiple direct LOS rays can coexist if they hit different patches of the sphere.
        key = (r.transmitter_id, r.n_bounces, *np.round(r.points[-1], dec))
        if key not in seen:
            seen.add(key)
            unique.append(r)
            
    return unique
