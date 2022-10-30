import ResNet_tuning

for lr in [1e-2, 1e-3, 1e-4, 1e-5]:
    for batch_size in [4,32,64]:
        for model_name in ['ResNet18', 'ResNet34', 'ResNet50']:
            ResNet_tuning.build_model(model_name, batch_size, lr)