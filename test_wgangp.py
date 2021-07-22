from models import WGANGP


if __name__ == '__main__':
    MODE =  'build'
    MAX_EPOCHS = 6000
    SHOW_EVERY_N = 600

    # Model
    model = WGANGP(
        data_name='mnist',
        ID=0,
        )
    model.summary()

    if MODE == 'load':
        model.load_models()

    # Load Data
    (x_train, y_train), _ = model.loader.load_mnist()

    # Train
    model.train(
        MAX_EPOCHS,
        SHOW_EVERY_N,
        train_data=[x_train, y_train]
        )
