import hlsvdpro as hlsvdpro
import mat73
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve


def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.csc_matrix(np.diff(np.eye(L), 2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

def watrem(data, dt, n, f):
    npts = len(data)
    dwell = dt
    nsv_sought = n
    result = hlsvdpro.hlsvd(data, nsv_sought, dwell)
    nsv_found, singvals, freq, damp, ampl, phas = result
    idx = np.where(np.abs(result[2]) < f)
    result = (len(idx),result[1],result[2][idx],result[3][idx],result[4][idx],result[5][idx])
    fid = hlsvdpro.create_hlsvd_fids(result, npts, dwell, sum_results=True, convert=False)

    # chop = ((((np.arange(len(fid)) + 1) % 2) * 2) - 1)
    # dat = data * chop
    # fit = fid * chop
    # dat[0] *= 0.5
    # fit[0] *= 0.5
    # BW = 1 / dt
    # fr = np.linspace(-BW / 2, BW / 2, 4096)
    # plt.plot(fr,np.fft.fftshift(np.fft.fft(data)).real, color='r')
    # plt.plot(fr,np.fft.fftshift(np.fft.fft(fid)).real, color='b')
    # plt.plot(fr,np.fft.fftshift(np.fft.fft(data-fid).real), color='g')
    # plt.show()
    return data - fid
def init(dataset, t_step,f):
    for idx in range(len(dataset[0])):
        dataset[:,idx] = watrem(dataset[:,idx],t_step,15,f)
        if idx % 100 == 0:
            print(str(idx))
    return dataset
# np.save(f, dataset, allow_pickle=True)
# with open('SfidsCoiledPRESS_WR.npy', 'rb') as f:
#     dataset = np.load(f)
# x = dataset[:,0]
# y = baseline_als(dataset[:,0],100, 0.005)

