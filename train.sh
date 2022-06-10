#Supervised Learning
python train_sup.py --save_path=results/sup --epoch=100

#Semi-supervised Learning
python noisy_student.py --save_path=results/noisy --epoch=100 --step=8 --threshold=0.8
python mpl.py --save_path=results/mpl --epoch=100 --threshold=0.8
python VAT.py --save_path=results/vat --epoch=100
python fixmatch.py --save_path=results/fix --epoch=100

#Self-supervised Learning
python AE.py --save_path=results/ae --epoch=100 --feature_dim=768
python DAE.py --save_path=results/ae --epoch=100 --feature_dim=768
python supcon.py --save_path=results/ae --epoch=100 --feature_dim=768 --temperature=0.7
python simclr.py --save_path=results/ae --epoch=100 --feature_dim=768 --temperature=0.7