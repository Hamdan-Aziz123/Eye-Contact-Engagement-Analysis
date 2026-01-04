Run command used (approx):

python ViT_train.py --processed_dir /content/project/data/processed --output_dir /content/drive/MyDrive/ResNet_Project/outputs_vit_model --image_size 224 --batch_size 32 --epochs 25 --freeze_epochs 5 --lr 0.0001 --fine_tune_lr 1e-05 --weight_decay 0.0001 --num_workers 2 --seed 42 --thr_att 10.0 --thr_dist 25.0 --model_name vit_base_patch16_224 --train_csv None --val_csv None --test_csv None --patience_lr 3

Files saved here:
- best_vit.pth : best model checkpoint (by val acc)
- training_history.csv : per-epoch metrics
- test_report.json : final test metrics (accuracy, precision, recall, f1, confusion matrix)
- classification_report.txt : sklearn classification_report text

Notes:
- Use --print_stats_only to print distribution stats for pitch/vert/horiz to help choose thresholds.
- No horizontal flip used because gaze/horiz would invert.
