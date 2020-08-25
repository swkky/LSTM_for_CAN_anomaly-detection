import tensorflow as tf
import os
import model_creator
import frame_reader_from_file
import bits_extractor
import dataset_creator_new
import numpy as np
import testing_dataset_creator
import logging
logging.getLogger('tensorflow').disabled = True

all_ids = ['0CF00400', '0CF00300', '18FEF100', '1CFF6F00', '18ECFF00', '18FF8800', '18FF8400',
           '18FEE500', '18F00029', '18FEF200', '18FF7F00', '1CFF7100', '18EBFF00', '18FF8200',
           '18FF8600', '18FEDC00', '1CFF7700', '18FF8900', '18FEDF00', '18FEE900', '18FF8700',
           '18FEE700', '1CFEB300', '18FEC100', '18FEEE00', '18ECFF29', '18EBFF29', '0C000027',
           '0C000F27', '18FEF111', '0CF00203', '0CF00327', '18FF8327', '0C002927', '18FF5027',
           '18F00503', '18FF5127', '18FEED11', '18FEE617', '1CFFAA27', '18EC0027', '18EB0027']


bs_size = 1
train_min_index=0
train_max_index = 300000
duration = 1
memory = 3000
num_layers = 1
embedding_size = 16
LSTM_units = 128

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
model = model_creator.my_model(bs_size, LSTM_units,embedding_size, num_layers)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

file = open("minatobus-candump-2019-05-08_030759.log")

_, train_data = frame_reader_from_file.prepare_dataset(file, min_index=train_min_index,
                                                       max_index=train_max_index, arbitration_id=all_ids[0])
train_data = train_data[:int(len(train_data) * 0.01)]

# bits extraction from packets for training
bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(
    np.array(train_data))


input_test_data, output_test_data = testing_dataset_creator.ready_for_training(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                    bit_8, bit_9,
                                                    bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16,
                                                    bit_17, bit_18,
                                                    bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25,
                                                    bit_26, bit_27,
                                                    bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34,
                                                    bit_35, bit_36,
                                                    bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43,
                                                    bit_44, bit_45,
                                                    bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52,
                                                    bit_53, bit_54,
                                                    bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61,
                                                    bit_62, bit_63,
                                                    batch_size=bs_size, duration=duration, memory_size=memory,
                                                    arbitration_id=all_ids[0])

def loss(labels, logits):
    return tf.keras.losses.binary_crossentropy(labels, logits)
print(input_test_data)
print(output_test_data)
a=0
counter=0
for inp, oup in zip(input_test_data.take(1).as_numpy_iterator(),output_test_data.take(1).as_numpy_iterator()):
    prediction=model.predict(inp)
    for pred, true_oup in zip(prediction, oup):
        counter=counter+1
        pre=pred.reshape(1,50)
        print(pre)
        a=a+ loss(true_oup,pre)
    print(a/counter)
    print(counter)
    break




