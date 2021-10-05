import os
import sys
from mmcv import Config


# root directory for dataset
_DS_ROOT = 'night'
# config
_CONFIG = 'rnw_rc'
# eval check points
_EVAL_CHECK_POINTS = [7, 11]


if __name__ == '__main__':
    assert len(_EVAL_CHECK_POINTS) > 0, 'Check points can not be empty.'
    config_path = f'configs/{_CONFIG}.yaml'
    curr_dir = os.path.abspath(os.curdir)
    cfg = Config.fromfile(config_path)
    for ck in _EVAL_CHECK_POINTS:
        result_file_name = os.path.join(curr_dir, f'checkpoints/{_CONFIG}/', 'rc_results/', 'result_{}.json'.format(ck))
        cmds = [
            'cd {}'.format(curr_dir),
            'python test_robotcar_disp.py {} {} {}'.format(_DS_ROOT, _CONFIG, f'checkpoints/{_CONFIG}/checkpoint_epoch={ck}.ckpt'),
            'cd evaluation',
            'python eval_robotcar.py {} --output_file_name={}'.format(_DS_ROOT, result_file_name),
        ]
        for cmd in cmds:
            if cmd.startswith('cd '):
                d = cmd.split()[-1]
                os.chdir(d)
            else:
                if os.system(cmd) != 0:
                    print('Errors encountered when executing command.')
                    sys.exit()
    os.chdir(curr_dir)
    os.system('python show_eval_result.py {} --title={}'.format(os.path.dirname(result_file_name),
                                                                os.path.basename(config_path)))
