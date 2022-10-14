# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
# python3 0.9_test_intent.py --test_file "${1}" --ckpt_path ckpt/intent/0.925_best.pt --pred_file "${2}"
python3 test_intent.py --test_file "${1}" --ckpt_path model/hw1_1_model.pt --pred_file "${2}"