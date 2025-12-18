import pycolmap
from pathlib import Path
import torch

from . import logger
from .triangulation import (
    estimation_and_geometric_verification,
    import_features,
    import_matches,
)


def create_empty_db(database_path: Path):
    if database_path.exists():
        logger.warning("The database already exists, deleting it.")
        database_path.unlink()
    logger.info("Creating an empty database...")
    with pycolmap.Database.open(database_path) as _:
        pass

@torch.no_grad()
def main(export_path: Path):
    database = export_path / "database.db"

    logger.info(f"Writing COLMAP logs to {sfm_dir / 'colmap.LOG.*'}")
    pycolmap.logging.set_log_destination(pycolmap.logging.INFO, sfm_dir / "colmap.LOG.")

    create_empty_db(database)
    import_images(image_dir, database, camera_mode, image_list, image_options)
    image_ids = get_image_ids(database)
    with pycolmap.Database.open(database) as db:
        import_features(image_ids, db, features)
        import_matches(
            image_ids,
            db,
            pairs,
            matches,
            min_match_score,
            skip_geometric_verification,
        )
    if not skip_geometric_verification:
        estimation_and_geometric_verification(database, pairs, verbose=True)


if __name__ == "__main__":
    export_database(Path("/home/eb-anands/exp/Beta-MiLO/Data/mihama/hloc/sfm/"))