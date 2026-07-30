from __future__ import annotations

import argparse
import sys
import hashlib
import json
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CACHE_DIR = Path("cache")

from src.core.precompute.precompute import precompute
from src.core.cache import save_scene
from src.core.scene.domain import Receiver
from script.urban_scene_gen import generate_urban_scene
from src.core.rx.apply_rx import apply_rx


@dataclass
class UrbanConfig:
    seed: int = 42

    domain_x: float = 500.0
    domain_y: float = 500.0

    block_size_x: float = 80.0
    block_size_y: float = 80.0

    street_width: float = 15.0
    alley_width: float = 7.0

    block_noise: float = 0.15
    arterial_prob: float = 0.20

    residential_frac: float = 0.65

    parcel_step_res: float = 20.0
    parcel_step_com: float = 40.0

    setback: float = 1.2
    density: float = 0.80

    h_res_mu: float = 5.0
    h_res_sigma: float = 1.5
    h_res_min: float = 3.0
    h_res_max: float = 12.0

    h_com_mu: float = 22.0
    h_com_sigma: float = 12.0
    h_com_min: float = 8.0
    h_com_max: float = 60.0

    tower_prob: float = 0.05
    lshape_prob: float = 0.15
    complex_prob: float = 0.05

    n_transmitters: int = 1
    tx_frequency: float = 1800e6
    tx_power_w: float = 40.0
    tx_height_offset: float = 20.0

    n_rays: int = 250_000
    batch_size: int = 100_000
    n_max: int = 10

    use_physics: bool = True
    temperature_c: float = 20.0
    bandwidth_hz: float = 8e6


def get_scene_hash(cfg: UrbanConfig) -> str:
    cfg_dict = {
        k: v
        for k, v in cfg.__dict__.items()
        if not k.startswith("_")
    }

    s = json.dumps(cfg_dict, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:10]


def build_street_rx_positions(
    cfg: UrbanConfig,
    z: float = 2.5,
):
    """
    Posiciones Rx determinísticas sobre el eje central de las calles.

    Incluye:
      - todas las intersecciones;
      - 1/4, 1/2 y 3/4 de cada cuadra sobre cada calle vertical;
      - 1/4, 1/2 y 3/4 de cada cuadra sobre cada calle horizontal.

    Los cuartos de cuadra se miden entre los bordes de dos calles
    transversales consecutivas. Por lo tanto, para una cuadra de
    80 metros, los puntos quedan a 20, 40 y 60 metros desde el
    comienzo de la cuadra.
    """
    W = cfg.domain_x
    H = cfg.domain_y

    pitch_x = cfg.block_size_x + cfg.street_width
    pitch_y = cfg.block_size_y + cfg.street_width

    # Se reproduce exactamente la misma cuenta que utiliza
    # generate_urban_scene para construir las calles.
    nx_blocks = int(W / pitch_x) + 1
    ny_blocks = int(H / pitch_y) + 1

    vertical_starts = [
        i * pitch_x
        for i in range(nx_blocks)
        if i * pitch_x < W
    ]

    horizontal_starts = [
        j * pitch_y
        for j in range(ny_blocks)
        if j * pitch_y < H
    ]

    # Centro real de cada franja de asfalto.
    #
    # El min también permite manejar una eventual calle recortada
    # por el límite del dominio.
    vertical_centers = [
        0.5 * (
            sx0
            + min(sx0 + cfg.street_width, W)
        )
        for sx0 in vertical_starts
    ]

    horizontal_centers = [
        0.5 * (
            sy0
            + min(sy0 + cfg.street_width, H)
        )
        for sy0 in horizontal_starts
    ]

    rx_positions = []

    # -----------------------------------------------------------------
    # 1. Todas las intersecciones
    # -----------------------------------------------------------------
    #
    # Una intersección se representa mediante el cruce entre el eje
    # central de una calle vertical y el eje central de una horizontal.
    for x in vertical_centers:
        for y in horizontal_centers:
            rx_positions.append(
                np.array(
                    [x, y, z],
                    dtype=float,
                )
            )

    fractions = (
        0.25,
        0.50,
        0.75,
    )

    # -----------------------------------------------------------------
    # 2. Cuarto, mitad y tres cuartos de cuadra sobre calles verticales
    # -----------------------------------------------------------------
    #
    # La coordenada x permanece en el centro de la calle vertical.
    # La coordenada y avanza dentro de cada cuadra.
    for x in vertical_centers:
        for j in range(len(horizontal_starts) - 1):
            # Fin de la calle transversal inferior.
            y0 = (
                horizontal_starts[j]
                + cfg.street_width
            )

            # Comienzo de la calle transversal superior.
            y1 = horizontal_starts[j + 1]

            for fraction in fractions:
                y = y0 + fraction * (y1 - y0)

                rx_positions.append(
                    np.array(
                        [x, y, z],
                        dtype=float,
                    )
                )

    # -----------------------------------------------------------------
    # 3. Cuarto, mitad y tres cuartos de cuadra sobre calles horizontales
    # -----------------------------------------------------------------
    #
    # La coordenada y permanece en el centro de la calle horizontal.
    # La coordenada x avanza dentro de cada cuadra.
    for y in horizontal_centers:
        for i in range(len(vertical_starts) - 1):
            # Fin de la calle transversal izquierda.
            x0 = (
                vertical_starts[i]
                + cfg.street_width
            )

            # Comienzo de la calle transversal derecha.
            x1 = vertical_starts[i + 1]

            for fraction in fractions:
                x = x0 + fraction * (x1 - x0)

                rx_positions.append(
                    np.array(
                        [x, y, z],
                        dtype=float,
                    )
                )

    # Eliminar posibles duplicados conservando el orden.
    # Con esta geometría no deberían existir, pero se deja la
    # protección para evitar problemas por redondeo.
    unique_positions = []
    seen = set()

    for position in rx_positions:
        key = tuple(np.round(position, 9))

        if key in seen:
            continue

        seen.add(key)
        unique_positions.append(position)

    return unique_positions


def extract_ray_features(
    ray,
    tx,
    scene_id,
    iteration_id,
    rx_pos,
    iter_seed,
    static_time_s,
    rx_time_s,
):
    if len(ray.points) < 2:
        return None

    last_bounce = ray.points[-2]
    n_bounces = len(ray.points) - 2

    lobe_id = ray.transmitter_id % 3
    lobe_theta_rad = (
        (2 * np.pi / 3)
        * lobe_id
    )

    return {
        "scene_id": str(scene_id),
        "iteration_id": int(iteration_id),
        "static_seed": int(iter_seed),
        "static_compute_time_s": float(static_time_s),
        "rx_compute_time_s": float(rx_time_s),

        "tau_s": float(ray.delay()),
        "azimuth_rad": float(ray.azimuth()),
        "elevation_rad": float(ray.elevation()),
        "doppler_hz": float(
            getattr(
                ray,
                "doppler_shift",
                0.0,
            )
        ),

        "freq_hz": float(ray.frequency),
        "power_dbm": float(ray.power_dbm),
        "n_bounces": int(n_bounces),

        "last_bounce_x": float(last_bounce[0]),
        "last_bounce_y": float(last_bounce[1]),
        "last_bounce_z": float(last_bounce[2]),

        "rx_pos_x": float(rx_pos[0]),
        "rx_pos_y": float(rx_pos[1]),
        "rx_pos_z": float(rx_pos[2]),

        "tx_pos_x": float(tx.position[0]),
        "tx_pos_y": float(tx.position[1]),
        "tx_pos_z": float(tx.position[2]),

        "tx_power_w": float(tx.tx_power_w),
        "tx_freq_hz": float(tx.frequency),
        "tx_lobe_theta_rad": float(lobe_theta_rad),
    }


def generate_dataset(
    base_seed: int,
    n_realizations: int,
    output_dir: Path,
):
    # La escena y el Tx se generan una única vez para este archivo.
    #
    # Todas las realizaciones reutilizan:
    #   - la misma geometría urbana;
    #   - el mismo transmisor;
    #   - las mismas posiciones de receptor.
    #
    # Lo que cambia entre realizaciones es el seed de precompute.
    cfg = UrbanConfig(
        seed=base_seed,
    )

    scene = generate_urban_scene(cfg)
    scene_id = get_scene_hash(cfg)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_scene(
        scene,
        output_dir / f"scene_{scene_id}.json",
    )

    # ================================================================
    # ÚNICO CAMBIO DE MUESTREO RESPECTO DEL BUILDER ANTERIOR
    # ================================================================
    #
    # Antes:
    #     rx_positions = sample_rx_positions(...)
    #
    # Ahora:
    #     todas las posiciones son determinísticas y están sobre calles.
    #
    # Se calculan una sola vez y se reutilizan en las diez realizaciones.
    rx_positions = build_street_rx_positions(cfg)

    print(
        f"Starting generation for scene {scene_id} "
        f"({n_realizations} realizations)..."
    )

    print(
        "  -> Number of deterministic street RX positions: "
        f"{len(rx_positions)}"
    )

    # Solamente se imprime la posición para poder verificar que
    # el Tx queda fijo durante todas las realizaciones.
    for tx_idx, tx in enumerate(scene.transmitters):
        print(
            f"  -> TX {tx_idx}: "
            f"x={tx.position[0]:.3f}, "
            f"y={tx.position[1]:.3f}, "
            f"z={tx.position[2]:.3f}"
        )

    dataset_rows = []

    for i in range(n_realizations):
        iter_seed = (
            base_seed
            + i
            + 1000
        )

        print(
            f"  -> Computing realization "
            f"{i + 1}/{n_realizations} "
            f"[Seed: {iter_seed}]"
        )

        # ------------------------------------------------------------
        # Static-field computation
        # ------------------------------------------------------------
        t0 = time.perf_counter()

        static_field = precompute(
            scene,
            seed=iter_seed,
            batch_size=cfg.batch_size,
        )

        static_time_s = (
            time.perf_counter()
            - t0
        )

        # ------------------------------------------------------------
        # Evaluar todas las posiciones Rx para esta realización
        # ------------------------------------------------------------
        for rx_pos in rx_positions:
            rx = Receiver(
                position=rx_pos,
                radius=10.0,
            )

            t1 = time.perf_counter()

            static_rx = apply_rx(
                static_field,
                rx,
            )

            rx_time_s = (
                time.perf_counter()
                - t1
            )

            rx_rays = getattr(
                static_rx,
                "anchors",
                [],
            )

            for ray in rx_rays:
                base_tx_idx = (
                    ray.transmitter_id
                    // 3
                )

                tx = scene.transmitters[
                    base_tx_idx
                ]

                features = extract_ray_features(
                    ray=ray,
                    tx=tx,
                    scene_id=scene_id,
                    iteration_id=i,
                    rx_pos=rx_pos,
                    iter_seed=iter_seed,
                    static_time_s=static_time_s,
                    rx_time_s=rx_time_s,
                )

                if features:
                    dataset_rows.append(
                        features
                    )

        del static_field

    # -----------------------------------------------------------------
    # Guardar todas las realizaciones de la escena en un único Parquet
    # -----------------------------------------------------------------
    if dataset_rows:
        df = pd.DataFrame(
            dataset_rows
        )

        parquet_path = (
            output_dir
            / f"dataset_{scene_id}.parquet"
        )

        df.to_parquet(
            parquet_path,
            engine="pyarrow",
            index=False,
        )

        print(
            f"\n[OK] Generated {len(df)} rays. "
            f"Dataset saved to: {parquet_path}"
        )

    else:
        print(
            "\n[WARNING] No rays reached the RX "
            "in any iteration."
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate an urban ray-tracing dataset using all "
            "street intersections and quarter/mid-block Rx positions."
        )
    )

    parser.add_argument(
        "--scene-seed",
        type=int,
        default=42,
        help=(
            "Seed used to generate the urban scene "
            "and its transmitter."
        ),
    )

    parser.add_argument(
        "--n-realizations",
        type=int,
        default=10,
        help=(
            "Number of static-field realizations generated "
            "for the same scene."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CACHE_DIR / "dataset_gamma",
        help=(
            "Directory in which scene JSON and Parquet files "
            "will be written."
        ),
    )

    # Compatibilidad con comandos anteriores.
    #
    # Se acepta --n-rx para que un comando viejo no falle, pero
    # el argumento se ignora deliberadamente: dataset_gamma siempre
    # evalúa todas las posiciones determinísticas.
    parser.add_argument(
        "--n-rx",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args(argv)

    if args.n_realizations <= 0:
        parser.error(
            "--n-realizations must be greater than zero"
        )

    if args.n_rx is not None:
        print(
            "[INFO] --n-rx is ignored: dataset_gamma always "
            "uses all deterministic street RX positions."
        )

    generate_dataset(
        base_seed=args.scene_seed,
        n_realizations=args.n_realizations,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()