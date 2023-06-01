import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import six
import shlex
import subprocess as sp
import tempfile


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def pad_if_too_small(data, sz, data_type=None):

    if len(data.shape) == 2:
        h, w = data.shape

        if not (h >= sz[0] and w >= sz[1]):
            new_h, new_w = max(h, sz[0]), max(w, sz[1])
            new_data = np.zeros([new_h, new_w], dtype=data_type if data_type else data.dtype)

            centre_h, centre_w = int(new_h / 2.), int(new_w / 2.)
            h_start, w_start = centre_h - int(h / 2.), centre_w - int(w / 2.)

            new_data[h_start:(h_start + h), w_start:(w_start + w)] = data
        else:
            new_data = data
    else:
        d, h, w = data.shape

        if not (d >= sz[0] and h >= sz[1] and w >= sz[2]):
            new_d, new_h, new_w = max(d, sz[0]), max(h, sz[1]), max(w, sz[2])
            new_data = np.zeros([new_d, new_h, new_w], dtype=data_type if data_type else data.dtype)

            centre_d, centre_h, centre_w = int(new_d /2.), int(new_h / 2.), int(new_w / 2.)
            d_start, h_start, w_start = centre_d - int(d / 2.), centre_h - int(h / 2.), centre_w - int(w / 2.)

            new_data[d_start:(d_start + d), h_start:(h_start + h), w_start:(w_start + w)] = data
        else:
            new_data = data

    return new_data

def pad_and_or_crop(orig_data, sz, data_type=None):
    data = pad_if_too_small(orig_data, sz, data_type)

    if len(data.shape) == 2:
        h, w = data.shape
        h_c = int(h / 2.)
        w_c = int(w / 2.)

        h_start = h_c - int(sz[0] / 2.)
        w_start = w_c - int(sz[1] / 2.)
        data = data[h_start:(h_start + sz[0]), w_start:(w_start + sz[1])]
    else:
        d, h, w = data.shape
        d_c = int(d / 2.)
        h_c = int(h / 2.)
        w_c = int(w / 2.)

        d_start = d_c - int(sz[0] / 2.)
        h_start = h_c - int(sz[1] / 2.)
        w_start = w_c - int(sz[2] / 2.)
        data = data[d_start:(d_start + sz[0]), h_start:(h_start + sz[1]), w_start:(w_start + sz[2])]

    return data


def imshow(img, title=''):
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.axis('off')
    plt.imshow(img, interpolation='none')
    plt.show()


def imsave(img, path):
    im = np.array(img.cpu()).astype(np.float64).squeeze()
    im /= np.max(im)
    im = Image.fromarray(im)
    im.save(path)


def _split_and_abspath_command(command, keep_string=False):
    is_string = False
    if isinstance(command, six.string_types):
        command = shlex.split(command)
        is_string = True
    if is_string and keep_string:
        command = ' '.join(command)
    return command


def check_call_out(
        command, cwd=None, print_exception=True, shell=False, ignore_exception=False, env=None):

    command = _split_and_abspath_command(command, shell)
    fpipe = tempfile.NamedTemporaryFile()

    try:
        if env:
            sp.check_call(command, cwd=cwd, shell=shell,
                          stdout=fpipe, stderr=sp.STDOUT, env=env)
        else:
            sp.check_call(command, cwd=cwd, shell=shell,
                          stdout=fpipe, stderr=sp.STDOUT)
        fpipe.seek(0)
        buf = fpipe.read()
    except OSError as e:
        if isinstance(command, six.string_types):
            command = shlex.split(command)
        e.filename = command[0]
        raise e
    except sp.CalledProcessError as e:
        fpipe.seek(0)
        buf = fpipe.read()

        if print_exception:
            print("Error: {error}".format(error=buf))
        e.output = buf
        if not ignore_exception:
            raise e
    finally:
        fpipe.close()

    return buf


def create_train_test_val_list(config):

    with open(config.IMAGE_FLIST, "r") as f:
        all_images = f.readlines()
    with open(config.LESION_FLIST, "r") as f:
        all_lesions = f.readlines()
    with open(config.BRAIN_FLIST, "r") as f:
        all_brain_masks = f.readlines()
    if config.XFM_FLIST:
        with open(config.XFM_FLIST, "r") as f:
            all_xfms = f.readlines()
    if config.TISSUE_FLIST:
        with open(config.TISSUE_FLIST, "r") as f:
            all_tissue_masks = f.readlines()

    config.DATA_COUNT = len(all_images)
    shuffle_list = random.sample(range(config.DATA_COUNT), k=config.DATA_COUNT)

    all_images = [all_images[i].rstrip() for i in shuffle_list]
    all_lesions = [all_lesions[i].rstrip() for i in shuffle_list]
    all_brain_masks = [all_brain_masks[i].rstrip() for i in shuffle_list]

    train_data_count = int(config.TRAIN_PCT * config.DATA_COUNT)
    val_data_count = int(config.VAL_PCT * config.DATA_COUNT)
    test_data_count = int(config.TEST_PCT * config.DATA_COUNT)

    images = [all_images[0:train_data_count], all_images[train_data_count:train_data_count+val_data_count],
              all_images[train_data_count+val_data_count:train_data_count+val_data_count+test_data_count]]
    lesions = [all_lesions[0:train_data_count], all_lesions[train_data_count:train_data_count+val_data_count],
               all_lesions[train_data_count+val_data_count:train_data_count+val_data_count+test_data_count]]
    brain_masks = [
        all_brain_masks[0:train_data_count], all_brain_masks[train_data_count:train_data_count+val_data_count],
        all_brain_masks[train_data_count+val_data_count:train_data_count+val_data_count+test_data_count]]
    if config.XFM_FLIST:
        all_xfms = [all_xfms[i].rstrip() for i in shuffle_list]
        xfms = [
            all_xfms[0:train_data_count], all_xfms[train_data_count:train_data_count+val_data_count],
            all_xfms[train_data_count+val_data_count:train_data_count+val_data_count+test_data_count]]
    if config.TISSUE_FLIST:
        all_tissue_masks = [all_tissue_masks[i].rstrip() for i in shuffle_list]
        tissue_masks = [
            all_tissue_masks[0:train_data_count], all_tissue_masks[train_data_count:train_data_count+val_data_count],
            all_tissue_masks[train_data_count+val_data_count:train_data_count+val_data_count+test_data_count]]
    else:
        tissue_masks = [None, None, None]
    if config.XFM_FLIST:
        return images, lesions, brain_masks, xfms, tissue_masks
    else:
        return images, lesions, brain_masks, tissue_masks

class EarlyStopping():
    def __init__(self, tolerance=20, min_delta=0):

        self.best_val_loss = np.Inf
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss > self.best_val_loss:
            self.counter +=1
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.counter = 0
            self.best_val_loss = val_loss

class Progbar(object):
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=25, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)
