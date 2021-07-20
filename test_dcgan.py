from models import DCGAN 


if __name__ == '__main__':
    MODE =  'build'
    BATCH_SIZE = 64
    MAX_EPOCHS = 6000
    SHOW_EVERY_N = 600

    # Model
    model = DCGAN(
        data_name='mnist',
        ID=0,
        )
    model.summary()
    if MODE == 'load':
        model.load_models()

    # Load Data
    (x_train, y_train), (x_test, y_test) = model.loader.load_np_data()

    # Train
    model.train(
        BATCH_SIZE,
        MAX_EPOCHS,
        SHOW_EVERY_N,
        train_data=[x_train, y_train]
        )
