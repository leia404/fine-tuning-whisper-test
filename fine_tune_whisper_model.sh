cd ../../model_ckpts/
mkdir fine-tuning_whisperNO_v1
cd fine-tuning_whisperNO_v1
mkdir runs
cd ../../fine_tuned_models/
mkdir whisperNO_v1
cd ../github/fine-tuning-whisper-NO/

python3 fine_tuning_whisperNO.py --original_model="openai/whisper-small"\
                                 --fine_tuned_model_ver="fine-tuning_whisperNO_v1"\
                                 --export_model_dir="../../fine_tuned_models/whisperNO_v1/"\
                                 --num_train_epochs=12\
                                 --learning_rate=5e-4\
                                 --use_asd_metric=0