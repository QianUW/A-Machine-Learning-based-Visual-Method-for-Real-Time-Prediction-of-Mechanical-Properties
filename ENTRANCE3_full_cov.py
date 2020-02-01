import os
import config as cfg
import full_cov


def pre_edit_cfg():
    if os.path.exists('./config_log.txt'):
        try:
            print('opening config log')
            f = open('./config_log.txt')
            lines = f.readlines()
            cfg.BATCH_SIZE = int(lines[0].strip('\n'))

            cfg.TRUNCATES_MEAN = float(lines[1].strip('\n'))
            cfg.TRUNCATES_STTDEV = float(lines[2].strip('\n'))
            cfg.BIAS = float(lines[3].strip('\n'))

            cfg.LOSS_PRINT_STEPS = int(lines[4].strip('\n'))

            cfg.TRAIN_STEP = float(lines[5].strip('\n'))
            cfg.EPOCHS = int(lines[6].strip('\n'))

            cfg.SAVE_STEPS = int(lines[7].strip('\n'))
            cfg.IS_RESTORE = lines[8].strip('\n') == 'True'
            cfg.SAVED_WEIGHT_STEPS = int(lines[9].strip('\n'))
            cfg.TEST_MODE = lines[10].strip('\n')
        except Exception as e:
            print(Exception,':', e)
            print('Config value error.Please resset.')
    cfg.print_setting()

if __name__ == '__main__':
    is_start = False
    is_pause = False
    pre_edit_cfg()
    dgnet = full_cov.diguinet()
    dgnet.start()
    is_start = True
    while True:
        command = input("Enter your command: ")
        if command == 'help':
            print('help,start,pause,resume,stop,savew,set,exit')
        elif command == 'weight_file':
            for wf in os.walk('./weight'):
                print(wf)
        elif command == 'start' and (not is_start):
            dgnet = full_cov.diguinet()
            dgnet.start()
            is_start = True
        elif command == 'pause' and is_start:
            dgnet.pause()
            is_pause = True
        elif command == 'resume' and is_pause:
            dgnet.resume()
            is_pause = False
        elif command == 'stop' and not is_pause and is_start:
            dgnet.stop()
            is_start = False
        elif command == 'savew' and is_start:
            dgnet.save()
        elif command == 'set':
            config_order = ['BATCH_SIZE','TRUNCATES_MEAN','TRUNCATES_STTDEV','BIAS','LOSS_PRINT_STEPS','TRAIN_STEP','EPOCHS','SAVE_STEPS','IS_RESTORE','SAVED_WEIGHT_STEPS','TEST_MODE']
            config_dict = {'BATCH_SIZE':cfg. BATCH_SIZE,'TRUNCATES_MEAN': cfg.TRUNCATES_MEAN,'TRUNCATES_STTDEV': cfg.TRUNCATES_STTDEV,'BIAS':cfg.BIAS,'LOSS_PRINT_STEPS': cfg.LOSS_PRINT_STEPS,'TRAIN_STEP':cfg.TRAIN_STEP,'EPOCHS':cfg.EPOCHS,'SAVE_STEPS': cfg.SAVE_STEPS,'IS_RESTORE': cfg.IS_RESTORE,'SAVED_WEIGHT_STEPS':cfg.SAVED_WEIGHT_STEPS,'TEST_MODE':cfg.TEST_MODE}
            cfg.print_setting()
            print('Saved and esc: save,Cancel and esc:cancel')
            while True:
                flag = False
                param_name = input('Enter the name of param:')
                if param_name == 'save':
                    f = open('./config_log.txt','w')
                    for name in config_order:
                        f.write(str(config_dict[name])+'\n')
                    f.close()
                    pre_edit_cfg()
                    print('The setting has been uppdated!')
                    break
                elif param_name == 'cancel':
                    print('canceled')
                    break
                elif not (param_name in config_dict):
                    print(param_name,'is not a param.')
                    continue
                param_value = input('Enter the value of param:')
                if param_name == 'save':
                    f = open('./config_log.txt','w')
                    for name in config_order:
                        f.write(str(config_dict[name])+'\n')
                    f.close()
                    pre_edit_cfg()
                    print('The setting has been uppdated!')
                    break
                elif param_name == 'cancel':
                    print('canceled')
                    break
                else:
                    config_dict[param_name] = param_value

        elif command == 'exit':
            os._exit(0)
        else:
            print('Thers is no command '+command)

