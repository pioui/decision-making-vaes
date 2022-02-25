import pickle


with open('trento-relaxed_nparticules_30.pkl', 'rb') as f:
    data = pickle.load(f)

print(data[['LOSS_GEN', 'LOSS_WVAR','MODEL_NAME','M_ACCURACY','M_ACCURACY_IS', 'ENTROPY', 'encoder_type' ]])