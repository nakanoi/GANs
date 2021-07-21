from models import CycleGAN


if __name__ == '__main__':
    # Hyper Paramators
    MODE =  'build'
    IMG_SIZE = 256
    TEST_A = 'n07740461_14740.jpg'
    TEST_B = 'n07749192_4241.jpg'
    BATCH_SIZE = 1
    MAX_EPOCHS = 200
    SHOW_EVERY_N = 10

    # Model & Train
    model = CycleGAN(
        input_dim=(IMG_SIZE, IMG_SIZE, 3),
        data_name='apple2orange',
        ID=0,
        )
    model.summary()

    if MODE == 'build':
        model.load_models()

    model.train(
        BATCH_SIZE,
        MAX_EPOCHS,
        SHOW_EVERY_N,
        test_data=[TEST_A, TEST_B],
        train_batch=True,
        )

