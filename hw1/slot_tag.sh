# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
python3 test_slot.py --data_dir "${1}" --ckpt_dir model/hw1_2_model.pt --pred_dir "${2}"
# python3 test_slot.py --data_dir "${1}" --ckpt_dir hw1_2_model/hw1_2_30_js_0.774.pt --pred_dir "${2}"