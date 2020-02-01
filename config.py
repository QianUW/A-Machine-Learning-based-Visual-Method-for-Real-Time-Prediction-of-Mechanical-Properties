BATCH_SIZE = 3

TRUNCATES_MEAN =0.0
TRUNCATES_STTDEV = 0.01
BIAS = 0.1

IMG_SIZE = (512,512)
LABEL_SIZE = (256,256)

LOSS_PRINT_STEPS = 20

TRAIN_STEP = 4.285e-06
EPOCHS = 200000

SAVE_STEPS = 5000
IS_RESTORE = False
SAVED_WEIGHT_STEPS = 1

TEST_MODE = 'on'

def print_setting():
    print('Current setting:')
    print( 'BATCH_SIZE:', BATCH_SIZE)
    print( '\n')
    print( 'TRUNCATES_MEAN:', TRUNCATES_MEAN)
    print( 'TRUNCATES_STTDEV:', TRUNCATES_STTDEV)
    print( 'BIAS:', BIAS)
    print( '\n')
    print( 'LOSS_PRINT_STEPS:', LOSS_PRINT_STEPS)
    print( '\n')
    print( 'TRAIN_STEP:', TRAIN_STEP)
    print( 'EPOCHS', EPOCHS)
    print( '\n')
    print( 'SAVE_STEPS', SAVE_STEPS)
    print( 'IS_RESTORE', IS_RESTORE)
    print( 'SAVED_WEIGHT_STEPS:', SAVED_WEIGHT_STEPS)
    print( 'TEST_MODE:', TEST_MODE)