
task_name=MRPC
seed=112401

echo "${task_name}"
echo "${seed}"

CUDA_VISIBLE_DEVICES="0" nohup python -u examples/research_projects/pabee_with_entropy/run_glue_with_pabee.py \
    --seed ${seed} \
    --task_name $task_name \
    --data_dir datasets/glue/${task_name}/ \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --output_dir experiments/pabee_with_entropy/outputs/bert_${task_name}_${seed} \
    --max_seq_length 128 \
    --do_train --do_eval \
    --evaluate_during_training \
    --do_lower_case \
    --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 64 \
    --learning_rate 2e-5 \
    --linear_learning_rate 2e-5 \
    --num_train_epochs 15 \
    --warmup_steps 200 --logging_steps 100 --save_steps 100 \
    --eval_all_checkpoints \
    --overwrite_output_dir --overwrite_cache \
    --gradient_checkpointing \
    --weights_schema asc \
    --patience -1 \
    --ee_mechanism V1 \
    --metric_key acc_and_f1 \
    "$@" > experiments/pabee_with_entropy/logs/bert_${task_name}_${seed}.nohup_log &

# CUDA_VISIBLE_DEVICES="0" nohup python examples/research_projects/bert_loses_patience/run_glue_with_pabee.py --model_type bert --model_name_or_path resources/bert/bert-base-uncased --task_name MRPC --do_train --do_eval --do_lower_case --data_dir datasets/glue/MRPC --max_seq_length 128 --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 --learning_rate 2e-5 --save_steps 400 --logging_steps 400 --num_train_epochs 15 --output_dir experiments/outputs/MRPC_bert_base_pabee_0110_2 --evaluate_during_training --gradient_checkpointing --overwrite_output_dir --patience -1 --weights_schema uniform > experiments/logs/MRPC_bert_base_pabee_0110_2.log &

# python examples/research_projects/pabee_with_entropy/run_glue_with_pabee.py --model_type bert --model_name_or_path resources/bert/bert-base-uncased --task_name MRPC --do_train --do_eval --do_lower_case --data_dir datasets/glue/MRPC --max_seq_length 128 --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 32 --learning_rate 2e-5 --linear_learning_rate 2e-5 --save_steps 100 --logging_steps 100 --num_train_epochs 10 --output_dir experiments/outputs/MRPC_bert_base_pabee_0407_0 --evaluate_during_training --gradient_checkpointing --overwrite_output_dir --overwrite_cache --patience -1 --weights_schema asc --metric_key avg-acc_and_f1 --patience -1 --ee_mechanism V1

## weights_schema = asc
# --learning_rate 2e-5 --linear_learning_rate 2e-4， 不使用BERT的pooler：avg-acc_and_f1 = 0.8194646923936874
# --learning_rate 1e-5 --linear_learning_rate 1e-4， 不使用BERT的pooler：avg-acc_and_f1 = 0.8187723809148163
# --learning_rate 2e-5 --linear_learning_rate 2e-5， 不使用BERT的pooler：avg-acc_and_f1 = 0.8265618310481155
# --learning_rate 5e-5 --linear_learning_rate 5e-5， 不使用BERT的pooler：avg-acc_and_f1 = 0.8229169420451629
# --learning_rate 2e-5 --linear_learning_rate 5e-4， 不使用BERT的pooler：avg-acc_and_f1 = 0.8228017676701969

***** Eval results %s *****
 0_acc = 0.6838235294117647
 0_acc_and_f1 = 0.7480253018237863
 0_entropy = tensor(0.8943, device='cuda:0')
 0_f1 = 0.8122270742358079
 10_acc = 0.8406862745098039
 10_acc_and_f1 = 0.8655370664285951
 10_entropy = tensor(0.0487, device='cuda:0')
 10_f1 = 0.8903878583473862
 11_acc = 0.8406862745098039
 11_acc_and_f1 = 0.8655370664285951
 11_entropy = tensor(0.0445, device='cuda:0')
 11_f1 = 0.8903878583473862
 1_acc = 0.6887254901960784
 1_acc_and_f1 = 0.7516620151710319
 1_entropy = tensor(0.8721, device='cuda:0')
 1_f1 = 0.8145985401459854
 2_acc = 0.696078431372549
 2_acc_and_f1 = 0.756593787957661
 2_entropy = tensor(0.8630, device='cuda:0')
 2_f1 = 0.8171091445427728
 3_acc = 0.7549019607843137
 3_acc_and_f1 = 0.7954837672774028
 3_entropy = tensor(0.7338, device='cuda:0')
 3_f1 = 0.8360655737704918
 4_acc = 0.7843137254901961
 4_acc_and_f1 = 0.8157679738562091
 4_entropy = tensor(0.4096, device='cuda:0')
 4_f1 = 0.8472222222222223
 5_acc = 0.8235294117647058
 5_acc_and_f1 = 0.8505402160864345
 5_entropy = tensor(0.1837, device='cuda:0')
 5_f1 = 0.8775510204081631
 6_acc = 0.821078431372549
 6_acc_and_f1 = 0.8489877822967298
 6_entropy = tensor(0.1129, device='cuda:0')
 6_f1 = 0.8768971332209106
 7_acc = 0.8308823529411765
 7_acc_and_f1 = 0.8566677522797876
 7_entropy = tensor(0.0895, device='cuda:0')
 7_f1 = 0.8824531516183987
 8_acc = 0.8431372549019608
 8_acc_and_f1 = 0.8673313393153872
 8_entropy = tensor(0.0675, device='cuda:0')
 8_f1 = 0.8915254237288135
 9_acc = 0.8431372549019608
 9_acc_and_f1 = 0.8673313393153872
 9_entropy = tensor(0.0549, device='cuda:0')
 avg-acc = 0.7875816993464051
 avg-acc_and_f1 = 0.824122117353084
 avg-f1 = 0.8606625353597627
 loss = 0.7719911222274487
 max-acc = 0.8431372549019608
 max-acc_and_f1 = 0.8673313393153872
 max-f1 = 0.8915254237288135
 std-acc = 0.061975833953164916
 std-acc_and_f1 = 0.04657531458748676
 std-f1 = 0.031419573782296546


## weights_schema = desc

***** Eval results %s *****
 0_acc = 0.6838235294117647
 0_acc_and_f1 = 0.7480253018237863
 0_entropy = tensor(0.8896, device='cuda:0')
 0_f1 = 0.8122270742358079
 10_acc = 0.8382352941176471
 10_acc_and_f1 = 0.8612229102167183
 10_entropy = tensor(0.2447, device='cuda:0')
 10_f1 = 0.8842105263157894
 11_acc = 0.8357843137254902
 11_acc_and_f1 = 0.8590169371790897
 11_entropy = tensor(0.2328, device='cuda:0')
 11_f1 = 0.882249560632689
 1_acc = 0.696078431372549
 1_acc_and_f1 = 0.752063983488132
 1_entropy = tensor(0.8047, device='cuda:0')
 1_f1 = 0.8080495356037151
 2_acc = 0.7034313725490197
 2_acc_and_f1 = 0.7584953472914591
 2_entropy = tensor(0.7655, device='cuda:0')
 2_f1 = 0.8135593220338984
 3_acc = 0.7524509803921569
 3_acc_and_f1 = 0.7933026658939437
 3_entropy = tensor(0.6442, device='cuda:0')
 3_f1 = 0.8341543513957307
 4_acc = 0.7794117647058824
 4_acc_and_f1 = 0.8115808823529411
 4_entropy = tensor(0.5045, device='cuda:0')
 4_f1 = 0.84375
 5_acc = 0.8112745098039216
 5_acc_and_f1 = 0.8382116857250781
 5_entropy = tensor(0.3925, device='cuda:0')
 5_f1 = 0.8651488616462346
 6_acc = 0.8259803921568627
 6_acc_and_f1 = 0.8510355712965815
 6_entropy = tensor(0.3220, device='cuda:0')
 6_f1 = 0.8760907504363002
 7_acc = 0.8333333333333334
 7_acc_and_f1 = 0.8570175438596492
 7_entropy = tensor(0.2920, device='cuda:0')
 7_f1 = 0.880701754385965
 8_acc = 0.8431372549019608
 8_acc_and_f1 = 0.8656245715069245
 8_entropy = tensor(0.2690, device='cuda:0')
 8_f1 = 0.8881118881118882
 9_acc = 0.8382352941176471
 9_acc_and_f1 = 0.8616263578602172
 9_entropy = tensor(0.2542, device='cuda:0')
 9_f1 = 0.8850174216027874
 avg-acc = 0.7867647058823529
 avg-acc_and_f1 = 0.8214353132078768
 avg-f1 = 0.8561059205334004
 loss = 0.45660504698753357
 max-acc = 0.8431372549019608
 max-acc_and_f1 = 0.8656245715069245
 max-f1 = 0.8881118881118882
 std-acc = 0.0592897510020226
 std-acc_and_f1 = 0.044757276100885056
 std-f1 = 0.03041280073189476


## weights_schema = uniform

***** Eval results %s *****
 0_acc = 0.6838235294117647
 0_acc_and_f1 = 0.7480253018237863
 0_entropy = tensor(0.8790, device='cuda:0')
 0_f1 = 0.8122270742358079
 10_acc = 0.8553921568627451
 10_acc_and_f1 = 0.8765695619669012
 10_entropy = tensor(0.2298, device='cuda:0')
 10_f1 = 0.8977469670710572
 11_acc = 0.8529411764705882
 11_acc_and_f1 = 0.8747464503042597
 11_entropy = tensor(0.2064, device='cuda:0')
 11_f1 = 0.896551724137931
 1_acc = 0.7009803921568627
 1_acc_and_f1 = 0.7583451809726911
 1_entropy = tensor(0.8791, device='cuda:0')
 1_f1 = 0.8157099697885196
 2_acc = 0.6936274509803921
 2_acc_and_f1 = 0.7493098253341898
 2_entropy = tensor(0.8889, device='cuda:0')
 2_f1 = 0.8049921996879874
 3_acc = 0.7426470588235294
 3_acc_and_f1 = 0.7865093129335741
 3_entropy = tensor(0.8387, device='cuda:0')
 3_f1 = 0.8303715670436188
 4_acc = 0.7671568627450981
 4_acc_and_f1 = 0.8050660346783343
 4_entropy = tensor(0.6597, device='cuda:0')
 4_f1 = 0.8429752066115703
 5_acc = 0.8112745098039216
 5_acc_and_f1 = 0.8402722294350677
 5_entropy = tensor(0.4496, device='cuda:0')
 5_f1 = 0.8692699490662139
 6_acc = 0.821078431372549
 6_acc_and_f1 = 0.8487794864138549
 6_entropy = tensor(0.3302, device='cuda:0')
 6_f1 = 0.8764805414551609
 7_acc = 0.8284313725490197
 7_acc_and_f1 = 0.8542841794251947
 7_entropy = tensor(0.2929, device='cuda:0')
 7_f1 = 0.8801369863013697
 8_acc = 0.8455882352941176
 8_acc_and_f1 = 0.8685772501771793
 8_entropy = tensor(0.2588, device='cuda:0')
 8_f1 = 0.891566265060241
 9_acc = 0.8504901960784313
 9_acc_and_f1 = 0.8727494009652054
 9_entropy = tensor(0.2340, device='cuda:0')
 9_f1 = 0.8950086058519794
 avg-acc = 0.7877859477124183
 avg-acc_and_f1 = 0.8236028512025199
 avg-f1 = 0.8594197546926213
 loss = 0.5151360608064212
 max-acc = 0.8553921568627451
 max-acc_and_f1 = 0.8765695619669012
 max-f1 = 0.8977469670710572
 std-acc = 0.06394865973207313
 std-acc_and_f1 = 0.04909329762678946
 std-f1 = 0.03437452639074747


####
# RTE
####

python examples/research_projects/pabee_with_entropy/run_glue_with_pabee.py --model_type bert --model_name_or_path resources/bert/bert-base-uncased --task_name RTE --do_train --do_eval --do_lower_case --data_dir datasets/glue/RTE --max_seq_length 128 --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 32 --learning_rate 5e-5 --linear_learning_rate 5e-5 --save_steps 100 --logging_steps 100 --num_train_epochs 8 --output_dir experiments/outputs/RTE_bert_base_pabee_0407_0 --evaluate_during_training --gradient_checkpointing --overwrite_output_dir --overwrite_cache --patience -1 --weights_schema uniform --metric_key avg-acc --patience -1 --ee_mechanism V1

## weights_schema = asc
***** Eval results %s *****
 0_acc = 0.5342960288808665
 0_entropy = tensor(0.9998, device='cuda:0')
 1_acc = 0.4981949458483754
 1_entropy = tensor(0.9991, device='cuda:0')
 2_acc = 0.5451263537906137
 2_entropy = tensor(0.9978, device='cuda:0')
 3_acc = 0.6425992779783394
 3_entropy = tensor(0.4816, device='cuda:0')
 4_acc = 0.6570397111913358
 4_entropy = tensor(0.3049, device='cuda:0')
 5_acc = 0.6570397111913358
 5_entropy = tensor(0.2017, device='cuda:0')
 6_acc = 0.6642599277978339
 6_entropy = tensor(0.1687, device='cuda:0')
 7_acc = 0.6570397111913358
 7_entropy = tensor(0.1531, device='cuda:0')
 8_acc = 0.6570397111913358
 8_entropy = tensor(0.1449, device='cuda:0')
 9_acc = 0.6642599277978339
 9_entropy = tensor(0.1465, device='cuda:0')
 10_acc = 0.6678700361010831
 10_entropy = tensor(0.1284, device='cuda:0')
 11_acc = 0.6606498194945848
 11_entropy = tensor(0.1138, device='cuda:0')
 avg-acc = 0.625451263537906
 loss = 1.5028889642821417
 max-acc = 0.6678700361010831
 std-acc = 0.058659631401227236


## weights_schema = desc

***** Eval results %s *****
 0_acc = 0.5342960288808665
 0_entropy = tensor(0.9998, device='cuda:0')
 10_acc = 0.6462093862815884
 10_entropy = tensor(0.6445, device='cuda:0')
 11_acc = 0.6498194945848376
 11_entropy = tensor(0.6487, device='cuda:0')
 1_acc = 0.51985559566787
 1_entropy = tensor(0.9985, device='cuda:0')
 2_acc = 0.5270758122743683
 2_entropy = tensor(0.9962, device='cuda:0')
 3_acc = 0.6389891696750902
 3_entropy = tensor(0.8274, device='cuda:0')
 4_acc = 0.6534296028880866
 4_entropy = tensor(0.7440, device='cuda:0')
 5_acc = 0.6534296028880866
 5_entropy = tensor(0.6846, device='cuda:0')
 6_acc = 0.6642599277978339
 6_entropy = tensor(0.6707, device='cuda:0')
 7_acc = 0.6534296028880866
 7_entropy = tensor(0.6607, device='cuda:0')
 8_acc = 0.6606498194945848
 8_entropy = tensor(0.6528, device='cuda:0')
 9_acc = 0.6425992779783394
 9_entropy = tensor(0.6436, device='cuda:0')
 avg-acc = 0.6203369434416366
 loss = 0.591713160276413
 max-acc = 0.6642599277978339
 std-acc = 0.05432849841206893

## weights_schema = uniform

***** Eval results %s *****
 0_acc = 0.5342960288808665
 0_entropy = tensor(0.9996, device='cuda:0')
 10_acc = 0.6462093862815884
 10_entropy = tensor(0.1313, device='cuda:0')
 11_acc = 0.6462093862815884
 11_entropy = tensor(0.1350, device='cuda:0')
 1_acc = 0.5379061371841155
 1_entropy = tensor(0.9961, device='cuda:0')
 2_acc = 0.5631768953068592
 2_entropy = tensor(0.9896, device='cuda:0')
 3_acc = 0.6462093862815884
 3_entropy = tensor(0.3836, device='cuda:0')
 4_acc = 0.6570397111913358
 4_entropy = tensor(0.2694, device='cuda:0')
 5_acc = 0.6714801444043321
 5_entropy = tensor(0.2123, device='cuda:0')
 6_acc = 0.6750902527075813
 6_entropy = tensor(0.1843, device='cuda:0')
 7_acc = 0.6606498194945848
 7_entropy = tensor(0.1689, device='cuda:0')
 8_acc = 0.6462093862815884
 8_entropy = tensor(0.1561, device='cuda:0')
 9_acc = 0.6498194945848376
 9_entropy = tensor(0.1482, device='cuda:0')
 avg-acc = 0.627858002406739
 loss = 1.2873156600528293
 max-acc = 0.6750902527075813
 std-acc = 0.049090799374203485

 ####
# CoLA
####

python examples/research_projects/pabee_with_entropy/run_glue_with_pabee.py --model_type bert --model_name_or_path resources/bert/bert-base-uncased --task_name CoLA --do_train --do_eval --do_lower_case --data_dir datasets/glue/CoLA --max_seq_length 128 --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 32 --learning_rate 2e-5 --linear_learning_rate 2e-5 --save_steps 100 --logging_steps 100 --num_train_epochs 8 --output_dir experiments/outputs/CoLA_bert_base_pabee_0407_0 --evaluate_during_training --gradient_checkpointing --overwrite_output_dir --overwrite_cache --patience -1 --weights_schema uniform --metric_key avg-mcc --patience -1 --ee_mechanism V1


***** Eval results %s *****
 0_entropy = tensor(0.8528, device='cuda:0')
 0_mcc = 0.0
 10_entropy = tensor(0.0453, device='cuda:0')
 10_mcc = 0.5200220944545176
 11_entropy = tensor(0.0423, device='cuda:0')
 11_mcc = 0.5325068699355778
 1_entropy = tensor(0.8575, device='cuda:0')
 1_mcc = 0.0
 2_entropy = tensor(0.8418, device='cuda:0')
 2_mcc = 0.0
 3_entropy = tensor(0.7551, device='cuda:0')
 3_mcc = 0.1694922489025642
 4_entropy = tensor(0.4153, device='cuda:0')
 4_mcc = 0.38474117687305953
 5_entropy = tensor(0.2186, device='cuda:0')
 5_mcc = 0.45077485078453133
 6_entropy = tensor(0.1168, device='cuda:0')
 6_mcc = 0.47896393100906826
 7_entropy = tensor(0.0813, device='cuda:0')
 7_mcc = 0.4894438128180558
 8_entropy = tensor(0.0628, device='cuda:0')
 8_mcc = 0.5179258445384731
 9_entropy = tensor(0.0498, device='cuda:0')
 9_mcc = 0.5279532549901665
 avg-mcc = 0.33931867369216784
 loss = 1.078967666084116
 max-mcc = 0.5325068699355778
 std-mcc = 0.2176073436612333


 ## weights_schema = desc

***** Eval results %s *****
 0_entropy = tensor(0.8382, device='cuda:0')
 0_mcc = 0.0
 10_entropy = tensor(0.1106, device='cuda:0')
 10_mcc = 0.5584013487121077
 11_entropy = tensor(0.1057, device='cuda:0')
 11_mcc = 0.5607990478394429
 1_entropy = tensor(0.8424, device='cuda:0')
 1_mcc = 0.013712849373137744
 2_entropy = tensor(0.8230, device='cuda:0')
 2_mcc = 0.07882828985982507
 3_entropy = tensor(0.6614, device='cuda:0')
 3_mcc = 0.2435297956527129
 4_entropy = tensor(0.3816, device='cuda:0')
 4_mcc = 0.33508616900126137
 5_entropy = tensor(0.2540, device='cuda:0')
 5_mcc = 0.386749909875835
 6_entropy = tensor(0.1900, device='cuda:0')
 6_mcc = 0.44949572497060086
 7_entropy = tensor(0.1515, device='cuda:0')
 7_mcc = 0.5072784045554821
 8_entropy = tensor(0.1312, device='cuda:0')
 8_mcc = 0.5633840892174439
 9_entropy = tensor(0.1141, device='cuda:0')
 9_mcc = 0.565965534490769
 avg-mcc = 0.3552692636290515
 loss = 0.5993886070721077
 max-mcc = 0.565965534490769
 std-mcc = 0.2114190119156135


## weights_schema = uniform

***** Eval results %s *****
 0_entropy = tensor(0.8534, device='cuda:0')
 0_mcc = 0.0
 10_entropy = tensor(0.0858, device='cuda:0')
 10_mcc = 0.5603228770167558
 11_entropy = tensor(0.0792, device='cuda:0')
 11_mcc = 0.5554433905906734
 1_entropy = tensor(0.8673, device='cuda:0')
 1_mcc = 0.0
 2_entropy = tensor(0.8638, device='cuda:0')
 2_mcc = 0.0
 3_entropy = tensor(0.7513, device='cuda:0')
 3_mcc = 0.18149047132447355
 4_entropy = tensor(0.4473, device='cuda:0')
 4_mcc = 0.37974526133988246
 5_entropy = tensor(0.2735, device='cuda:0')
 5_mcc = 0.4096847739402383
 6_entropy = tensor(0.1774, device='cuda:0')
 6_mcc = 0.4660859395741973
 7_entropy = tensor(0.1154, device='cuda:0')
 7_mcc = 0.4979366411311316
 8_entropy = tensor(0.1026, device='cuda:0')
 8_mcc = 0.5399823631367601
 9_entropy = tensor(0.0887, device='cuda:0')
 9_mcc = 0.5528474752734607
 avg-mcc = 0.3452949327772978
 loss = 0.7638277519832958
 max-mcc = 0.5603228770167558
 std-mcc = 0.2234429743047633
