import numpy
import os
import time


from nmt import train

def main(job_id, params):
    print ('timestamp {} {}'.format('running',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    print (params)
    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     n_words=params['n-words'][0],
                     n_words_src=params['n-words'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0],
                     patience=1000,
                     maxlen=50,
                     batch_size=80,
                     validFreq_fine=2000,
                     validFreq=10000,
                     val_burn_in=20000,
                     val_burn_in_fine=80000,
                     dispFreq=20,
                     saveFreq=2000,
                     sampleFreq=200,
                     datasets=['/home/ycli/resource/hw/ch.txt.shuffle',
                               '/home/ycli/resource/hw/en.txt.shuffle'],
                     valid_datasets=['/home/ycli/resource/hw/valid/valid_src',
                                     '/home/ycli/resource/hw/valid/valid_trg',
                                     './data/valid_out'],
                     dictionaries=['/home/ycli/resource/hw/vocab/vocab_src.pkl',
                                   '/home/ycli/resource/hw/vocab/vocab_trg.pkl'],
                     use_dropout=params['use-dropout'][0],
                     overwrite=False,
                     valid_mode=params['valid_mode'][0],
                     bleu_script=params['bleu_script'][0])
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['train_model.npz'],
        'dim_word': [620],
        'dim': [1000],
        'n-words': [30001],
        'optimizer': ['sgd'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [True],
        'learning-rate': [2.],
        'reload': [False],
        'valid_mode': ['bleu', 'ce'],
        'bleu_script': ['./data/mteval-v11b.pl', './data/multi-bleu.perl']})
