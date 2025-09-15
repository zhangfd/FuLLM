#!/usr/bin/env bash
# set -euo pipefail
# echo "Extracting predict data..."
# python scripts/03_data_postproc/extract_predict_data.py

# echo "Collecting source data..."
# python scripts/03_data_postproc/collect_source.py

# echo "Converting data for eval..."
# python scripts/03_data_postproc/convert_data_for_eval.py

echo "Eval..."
python scripts/eval.py --result_dir_name fig2_result --files fig2.xlsx
python scripts/eval.py --result_dir_name fig3_result --files fig3.xlsx
python scripts/eval.py --result_dir_name fig4_result --files fig4.xlsx
python scripts/eval.py --result_dir_name fig5_result --files fig5.xlsx
python scripts/eval.py --result_dir_name fig6_result --files fig6_part1.xlsx fig6_part2.xlsx
python scripts/eval.py --result_dir_name table2_result --files table2.xlsx
python scripts/eval.py --result_dir_name table3_result --files table3.xlsx
python scripts/eval.py --result_dir_name table4_result --files table4.xlsx
python scripts/eval.py --result_dir_name table5_result --files table5.xlsx
python scripts/eval.py --result_dir_name supp_table1_result --files supp_table1_part1.xlsx supp_table1_part2.xlsx supp_table1_part3.xlsx
python scripts/eval.py --result_dir_name supp_table2_result --files supp_table2_part1.xlsx supp_table2_part2.xlsx
python scripts/eval.py --result_dir_name supp_table3_result --files supp_table3.xlsx
python scripts/eval.py --result_dir_name supp_table4_result --files supp_table4_part1.xlsx supp_table4_part2.xlsx
