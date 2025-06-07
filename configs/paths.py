from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()


CROSSDOCKED = project_root / "crossdocked"

EXAMPLE_PROTEIN = (
    CROSSDOCKED
    / "crossdocked_pocket10"
    / "1A1C_MALDO_2_433_0"
    / "1m4n_A_rec_1m7y_ppg_lig_tt_min_0_pocket10.pdb"
)
