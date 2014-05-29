
import random
import signal
import os

from breze.learn.sgvb import VariationalAutoEncoder
from breze.learn.trainer.trainer import Trainer
from breze.learn.trainer.report import KeyPrinter, JsonPrinter
import climin.initialize
import numpy as np
from sklearn.grid_search import ParameterSampler


def preamble(i):
    train_folder = os.path.dirname(os.path.realpath(__file__))
    module = os.path.join(train_folder, 'vae_on_hands.py')
    script = '/nthome/maugust/git/alchemie/scripts/alc.py'
    runner = 'python %s run %s' % (script, module)

    pre = '#SUBMIT: runner=%s\n' % runner
    pre += '#SUBMIT: gpu=no\n'

    minutes_before_3_hour = 15
    slurm_preamble = '#SBATCH -J VAE_2hiddens_on_hands_%d\n' % (i)
    slurm_preamble += '#SBATCH --mem=15000\n'
    slurm_preamble += '#SBATCH --nice=1\n'
    slurm_preamble += '#SBATCH --signal=INT@%d\n' % (minutes_before_3_hour*60)
    slurm_preamble += '#SBATCH --exclude=cn-7,cn-8\n'
    return pre + slurm_preamble



def draw_pars(n=1):
    class OptimizerDistribution(object):
        def rvs(self):
            grid = {
                'step_rate': [0.0001, 0.0005, 0.005,0.001,0.00001,0.00005],
                'momentum': [0.99, 0.995,0.9,0.95],
                'decay': [0.9, 0.95,0.99],
            }

            sample = list(ParameterSampler(grid, n_iter=1))[0]
            sample.update({'step_rate_max': 0.05, 'step_rate_min': 1e-7})
            return 'rmsprop', sample

    grid = {
        'n_hiddens': [[[200,200],10,[200,200]],[[500,500],10,[500,500]],[[1000,1000],10,[1000,1000]],[[700,700],10,[700,700]],[[100,100],10,[100,100]],[[50,50],10,[50,50]]],
        'hidden_transfers': [[['sigmoid','sigmoid'],'identity',['sigmoid','sigmoid']], [['tanh','tanh'],'identity',['tanh','tanh']], [['rectifier','rectifier'],'identity',['rectifier','rectifier']]],
        'par_std': [1.5, 1, 1e-1, 1e-2,1e-3,1e-4,1e-5],
	    'batch_size': [10000,20000,50000,5000],
        'optimizer': OptimizerDistribution(),
    }

    sampler = ParameterSampler(grid, n)
    return sampler


def load_data(pars):
    X = np.loadtxt('../../hand_data_train_100Hz.txt')[:1600000]
    VX = np.loadtxt('../../hand_data_val_100Hz.txt')[:500000]
    #TX, TZ = np.loadtxt('hand_data_test_100Hz.txt')[:500000]

    return (X, VX), (X, VX)

def make_data_dict(trainer,data):
    train_data, val_data = data
    trainer.val_key = 'val'
    trainer.eval_data = {}
    trainer.eval_data['train'] = ([data for data in train_data])
    trainer.eval_data['val'] = ([data for data in val_data])


def new_trainer(pars, data):
    input_size = 54
    batch_size = pars['batch_size']
    m = VariationalAutoEncoder(input_size, pars['n_hiddens'][0], pars['n_hiddens'][1], pars['n_hiddens'][2],
            pars['hidden_transfers_encode'], pars['hidden_transfers_decode'], latent_prior='white_gauss',
            latent_posterior='diag_gauss', visible='diag_gauss', batch_size = batch_size,
            optimizer=pars['optimizer'])
    climin.initialize.randomize_normal(m.parameters.data, 0, pars['par_std'])

    n_report = 1600000/batch_size
    max_iter = n_report * 1000

    interrupt = climin.stops.OnSignal()
    print dir(climin.stops)
    stop = climin.stops.Any([
        climin.stops.AfterNIterations(max_iter),
        climin.stops.OnSignal(signal.SIGTERM),
        #climin.stops.NotBetterThanAfter(1e-1,500,key='train_loss'),
    ])

    pause = climin.stops.ModuloNIterations(n_report)
    reporter = KeyPrinter(['n_iter', 'val_loss'])

    t = Trainer(
        m,
        stop=stop, pause=pause, report=reporter,
        interrupt=interrupt)

    make_data_dict(t, data)

    return t


def make_report(pars, trainer, data):
    return {'train_loss': trainer.score(*trainer.eval_data['train']),
            'val_loss': trainer.score(*trainer.eval_data['val'])}

