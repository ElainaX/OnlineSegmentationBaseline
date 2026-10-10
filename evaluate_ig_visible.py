"""Original AP and confirmed event-grounded G-AP for the new IG experiment."""
from pathlib import Path
import recompute_standard_ap as standard
from eval_gap_v2 import run_highchange

standard.OUT=Path('/root/autodl-tmp/output/ig_visible_v3_20261010')
standard.ROOTS={'IGSubmap':standard.OUT/'detect','IGSubmap_delete':standard.OUT/'delete'}

if __name__=='__main__':
 standard.main()
 run_highchange(standard.OUT, standard.ROOTS)
