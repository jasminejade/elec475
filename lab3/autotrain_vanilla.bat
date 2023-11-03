:: #Set 1: Decrease lr parameter by a factor of 10
::start "" python vanilla.py --lr 0.01 --weight_decay 2e-03 --gamma 0.9 --momentum 0.9 --num_steps 0 --batch 150 --epochs 100 --encoder_pth "encoder.pth" --classifier_pth "autotest/vanilla/vanilla_classifier_1_1_.pth" --loss_plot "autotest/vanilla/loss.vanilla_1_1_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_1_1_.png" --training "y"

::start "" python vanilla.py --lr 0.001 --weight_decay 2e-03 --gamma 0.9 --momentum 0.9 --num_steps 0 --batch 150 --epochs 100 --encoder_pth "encoder.pth" --classifier_pth "autotest/vanilla/vanilla_classifier_1_2_.pth" --loss_plot "autotest/vanilla/loss.vanilla_1_2_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_1_2_.png" --training "y"

::start "" python vanilla.py --lr 0.0001 --weight_decay 2e-03 --gamma 0.9 --momentum 0.9 --num_steps 0 --batch 150 --epochs 100 --encoder_pth "encoder.pth" --classifier_pth "autotest/vanilla/vanilla_classifier_1_3_.pth" --loss_plot "autotest/vanilla/loss.vanilla_1_3_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_1_3_.png" --training "y"

::start "" python vanilla.py --lr 0.00001 --weight_decay 2e-03 --gamma 0.9 --momentum 0.9 --num_steps 0 --batch 150 --epochs 100 --encoder_pth "encoder.pth" --classifier_pth "autotest/vanilla/vanilla_classifier_1_4_.pth" --loss_plot "autotest/vanilla/loss.vanilla_1_4_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_1_4_.png" --training "y"

::timeout /t 6000 /nobreak

::# Set 2: Reduce weight_decay by a factor of 10
::start "" python vanilla.py --lr 0.1 --weight_decay 2e-04 --gamma 0.9 --momentum 0.9 --num_steps 0 --batch 150 --epochs 100 --encoder_pth "encoder.pth" --classifier_pth "autotest/vanilla/vanilla_classifier_2_1_.pth" --loss_plot "autotest/vanilla/loss.vanilla_2_1_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_2_1_.png" --training "y"

::start "" python vanilla.py --lr 0.1 --weight_decay 2e-05 --gamma 0.9 --momentum 0.9 --num_steps 0 --batch 150 --epochs 100 --encoder_pth "encoder.pth" --classifier_pth "autotest/vanilla/vanilla_classifier_2_2_.pth" --loss_plot "autotest/vanilla/loss.vanilla_2_2_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_2_2_.png" --training "y"

::start "" python vanilla.py --lr 0.1 --weight_decay 2e-06 --gamma 0.9 --momentum 0.9 --num_steps 0 --batch 150 --epochs 100 --encoder_pth "encoder.pth" --classifier_pth "autotest/vanilla/vanilla_classifier_2_3_.pth" --loss_plot "autotest/vanilla/loss.vanilla_2_3_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_2_3_.png" --training "y"

::start "" python vanilla.py --lr 0.1 --weight_decay 2e-07 --gamma 0.9 --momentum 0.9 --num_steps 0 --batch 150 --epochs 100 --encoder_pth "encoder.pth" --classifier_pth "autotest/vanilla/vanilla_classifier_2_4_.pth" --loss_plot "autotest/vanilla/loss.vanilla_2_4_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_2_4_.png" --training "y"

::timeout /t 6000 /nobreak

::# Set 3: Reduce gamma by 0.1
start "" python vanilla.py --lr 0.1 --weight_decay 2e-03 --gamma 0.8 --momentum 0.9 --num_steps 0 --batch 150 --epochs 100 --encoder_pth "encoder.pth" --classifier_pth "autotest/vanilla/vanilla_classifier_3_1_.pth" --loss_plot "autotest/vanilla/loss.vanilla_3_1_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_3_1_.png" --training "y"

start "" python vanilla.py --lr 0.1 --weight_decay 2e-03 --gamma 0.7 --momentum 0.9 --num_steps 0 --batch 150 --epochs 100 --encoder_pth "encoder.pth" --classifier_pth "autotest/vanilla/vanilla_classifier_3_2_.pth" --loss_plot "autotest/vanilla/loss.vanilla_3_2_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_3_2_.png" --training "y"

start "" python vanilla.py --lr 0.1 --weight_decay 2e-03 --gamma 0.6 --momentum 0.9 --num_steps 0 --batch 150 --epochs 100 --encoder_pth "encoder.pth" --classifier_pth "autotest/vanilla/vanilla_classifier_3_3_.pth" --loss_plot "autotest/vanilla/loss.vanilla_3_3_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_3_3_.png" --training "y"

start "" python vanilla.py --lr 0.1 --weight_decay 2e-03 --gamma 0.5 --momentum 0.9 --num_steps 0 --batch 150 --epochs 100 --encoder_pth "encoder.pth" --classifier_pth "autotest/vanilla/vanilla_classifier_3_4_.pth" --loss_plot "autotest/vanilla/loss.vanilla_3_4_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_3_4_.png" --training "y"

timeout /t 6000 /nobreak

::# Set 4: Reduce num_steps by 1
start "" python vanilla.py --lr 0.1 --weight_decay 2e-03 --gamma 0.9 --momentum 0.9 --num_steps 8 --batch 150 --epochs 150 --encoder_pth "encoder.pth" --classifier_pth "autotest/vanilla/vanilla_classifier_4_1_.pth" --loss_plot "autotest/vanilla/loss.vanilla_4_1_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_4_1_.png" --training "y"

start "" python vanilla.py --lr 0.1 --weight_decay 2e-03 --gamma 0.9 --momentum 0.9 --num_steps 7 --batch 150 --epochs 150 --encoder_pth "encoder.pth" --classifier_pth "autotest/vanilla/vanilla_classifier_4_2_.pth" --loss_plot "autotest/vanilla/loss.vanilla_4_2_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_4_2_.png" --training "y"

start "" python vanilla.py --lr 0.1 --weight_decay 2e-03 --gamma 0.9 --momentum 0.9 --num_steps 6 --batch 150 --epochs 150 --encoder_pth "encoder.pth" --classifier_pth "autotest/vanilla/vanilla_classifier_4_3_.pth" --loss_plot "autotest/vanilla/loss.vanilla_4_3_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_4_3_.png" --training "y"

start "" python vanilla.py --lr 0.1 --weight_decay 2e-03 --gamma 0.9 --momentum 0.9 --num_steps 5 --batch 150 --epochs 150 --encoder_pth "encoder.pth" --classifier_pth "autotest/vanilla/vanilla_classifier_4_4_.pth" --loss_plot "autotest/vanilla/loss.vanilla_4_4_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_4_4_.png" --training "y"

timeout /t 6000 /nobreak

::# Set 5: Reduce momentum by 0.1

::start "" python vanilla.py --lr 0.1 --weight_decay 2e-03 --gamma 0.9 --momentum 0.8 --num_steps 0 --batch 150 --epochs 100 --encoder_pth "encoder.pth" --classifier_pth "autotest/vanilla/vanilla_classifier_5_1_.pth" --loss_plot "autotest/vanilla/loss.vanilla_5_1_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_5_1_.png" --training "y"

::start "" python vanilla.py --lr 0.1 --weight_decay 2e-03 --gamma 0.9 --momentum 0.7 --num_steps 0 --batch 150 --epochs 100 --encoder_pth "encoder.pth" --classifier_pth "autotest/vanilla/vanilla_classifier_5_2_.pth" --loss_plot "autotest/vanilla/loss.vanilla_5_2_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_5_2_.png" --training "y"

::start "" python vanilla.py --lr 0.1 --weight_decay 2e-03 --gamma 0.9 --momentum 0.6 --num_steps 0 --batch 150 --epochs 100 --encoder_pth "encoder.pth" --classifier_pth "autotest/vanilla/vanilla_classifier_5_3_.pth" --loss_plot "autotest/vanilla/loss.vanilla_5_3_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_5_3_.png" --training "y"

::start "" python vanilla.py --lr 0.1 --weight_decay 2e-03 --gamma 0.9 --momentum 0.5 --num_steps 0 --batch 150 --epochs 100 --encoder_pth "encoder.pth" --classifier_pth "autotest/vanilla/vanilla_classifier_5_4_.pth" --loss_plot "autotest/vanilla/loss.vanilla_5_4_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_5_4_.png" --training "y"

:: # Test all

::timeout /t 6000 /nobreak

# Set 1
python vanilla.py --classifier_pth "autotest/vanilla/vanilla_classifier_1_1_.pth" --loss_plot "autotest/vanilla/loss.vanilla_1_1_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_1_1_.png" --training "n"
python vanilla.py --classifier_pth "autotest/vanilla/vanilla_classifier_1_2_.pth" --loss_plot "autotest/vanilla/loss.vanilla_1_2_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_1_2_.png" --training "n"
python vanilla.py --classifier_pth "autotest/vanilla/vanilla_classifier_1_3_.pth" --loss_plot "autotest/vanilla/loss.vanilla_1_3_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_1_3_.png" --training "n"
python vanilla.py --classifier_pth "autotest/vanilla/vanilla_classifier_1_4_.pth" --loss_plot "autotest/vanilla/loss.vanilla_1_4_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_1_4_.png" --training "n"

# Set 2
python vanilla.py --classifier_pth "autotest/vanilla/vanilla_classifier_2_1_.pth" --loss_plot "autotest/vanilla/loss.vanilla_2_1_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_2_1_.png" --training "n"
python vanilla.py --classifier_pth "autotest/vanilla/vanilla_classifier_2_2_.pth" --loss_plot "autotest/vanilla/loss.vanilla_2_2_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_2_2_.png" --training "n"
python vanilla.py --classifier_pth "autotest/vanilla/vanilla_classifier_2_3_.pth" --loss_plot "autotest/vanilla/loss.vanilla_2_3_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_2_3_.png" --training "n"
python vanilla.py --classifier_pth "autotest/vanilla/vanilla_classifier_2_4_.pth" --loss_plot "autotest/vanilla/loss.vanilla_2_4_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_2_4_.png" --training "n"

# Set 3
python vanilla.py --classifier_pth "autotest/vanilla/vanilla_classifier_3_1_.pth" --loss_plot "autotest/vanilla/loss.vanilla_3_1_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_3_1_.png" --training "n"
python vanilla.py --classifier_pth "autotest/vanilla/vanilla_classifier_3_2_.pth" --loss_plot "autotest/vanilla/loss.vanilla_3_2_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_3_2_.png" --training "n"
python vanilla.py --classifier_pth "autotest/vanilla/vanilla_classifier_3_3_.pth" --loss_plot "autotest/vanilla/loss.vanilla_3_3_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_3_3_.png" --training "n"
python vanilla.py --classifier_pth "autotest/vanilla/vanilla_classifier_3_4_.pth" --loss_plot "autotest/vanilla/loss.vanilla_3_4_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_3_4_.png" --training "n"

# Set 4
python vanilla.py --classifier_pth "autotest/vanilla/vanilla_classifier_4_1_.pth" --loss_plot "autotest/vanilla/loss.vanilla_4_1_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_4_1_.png" --training "n"
python vanilla.py --classifier_pth "autotest/vanilla/vanilla_classifier_4_2_.pth" --loss_plot "autotest/vanilla/loss.vanilla_4_2_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_4_2_.png" --training "n"
python vanilla.py --classifier_pth "autotest/vanilla/vanilla_classifier_4_3_.pth" --loss_plot "autotest/vanilla/loss.vanilla_4_3_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_4_3_.png" --training "n"
python vanilla.py --classifier_pth "autotest/vanilla/vanilla_classifier_4_4_.pth" --loss_plot "autotest/vanilla/loss.vanilla_4_4_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_4_4_.png" --training "n"

# Set 5
python vanilla.py --classifier_pth "autotest/vanilla/vanilla_classifier_5_1_.pth" --loss_plot "autotest/vanilla/loss.vanilla_5_1_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_5_1_.png" --training "n"
python vanilla.py --classifier_pth "autotest/vanilla/vanilla_classifier_5_2_.pth" --loss_plot "autotest/vanilla/loss.vanilla_5_2_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_5_2_.png" --training "n"
python vanilla.py --classifier_pth "autotest/vanilla/vanilla_classifier_5_3_.pth" --loss_plot "autotest/vanilla/loss.vanilla_5_3_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_5_3_.png" --training "n"
python vanilla.py --classifier_pth "autotest/vanilla/vanilla_classifier_5_4_.pth" --loss_plot "autotest/vanilla/loss.vanilla_5_4_.png" --accuracy_plot "autotest/vanilla/accuracy.vanilla_5_4_.png" --training "n"
