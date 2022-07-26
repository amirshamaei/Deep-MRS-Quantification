import numpy as np
def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def read(path):
    header = {}
    f = open(path, "r")
    i=0
    for x in f:
      step_0 = x.split(':')
      if step_0[0] == 'Filename':
          header['fn']=step_0[1].strip("\n")
      elif step_0[0] == 'PointsInDataset':
          header['PointsInDataset'] =int(step_0[1])
      elif step_0[0] == 'DatasetsInFile':
          header['DatasetsInFile'] =int(step_0[1])
      elif step_0[0] == 'SamplingInterval':
          header['t'] =float(step_0[1])
      elif step_0[0] == 'ZeroOrderPhase':
          header['ZeroOrderPhase'] = float(step_0[1])
      elif step_0[0] == 'BeginTime':
          header['BeginTime'] = float(step_0[1])
      elif step_0[0] == 'TransmitterFrequency':
          header['Transfreq'] = float(step_0[1])
      elif step_0[0] == 'MagneticField':
          header['MagneticField'] = float(step_0[1])
      elif step_0[0].startswith('sig(real)'):
          break
    re = np.zeros((header['DatasetsInFile']*header['PointsInDataset']),dtype="float32")
    imag = np.zeros((header['DatasetsInFile']*header['PointsInDataset']),dtype="float32")
    re_fr = np.zeros((header['DatasetsInFile']*header['PointsInDataset']),dtype="float32")
    imag_fr = np.zeros((header['DatasetsInFile']*header['PointsInDataset']),dtype="float32")
    for x in f:
        if isfloat(x.split("\t")[0]):
            re[i] = float(x.split("\t")[0])
            imag[i] = float(x.split("\t")[1])
            re_fr[i] = float(x.split("\t")[2])
            imag_fr[i] = float(x.split("\t")[3])
            i = i+1
    re= re.reshape((header['DatasetsInFile'],header['PointsInDataset']))
    imag= imag.reshape((header['DatasetsInFile'],header['PointsInDataset']))
    re_fr= re_fr.reshape((header['DatasetsInFile'],header['PointsInDataset']))
    imag_fr= imag_fr.reshape((header['DatasetsInFile'],header['PointsInDataset']))
    f.close()
    return header,re, imag, re_fr,imag_fr

def write(header, sig, path):
    sig = np.squeeze(sig)
    re = sig.real
    imag = sig.imag
    with open(path, 'w') as txtfile:
        txtfile.write('jMRUI Data Textfile\n')
        txtfile.write('\n')
        x = header['fn']
        txtfile.write(f'Filename: {x}\n')
        txtfile.write('\n')
        pd = header['PointsInDataset']
        txtfile.write(f'PointsInDataset: {pd}\n')
        df = header['DatasetsInFile']
        txtfile.write(f'DatasetsInFile: {df}\n')
        x = header['t']
        txtfile.write(f'SamplingInterval: {x}\n')
        x = header['ZeroOrderPhase']
        txtfile.write('ZeroOrderPhase: 0E0\n')
        x = header['BeginTime']
        txtfile.write('BeginTime: 0E0\n')
        x = header['Transfreq']
        txtfile.write(f'TransmitterFrequency: {x}\n')
        txtfile.write('MagneticField: 0E0\n')
        txtfile.write('TypeOfNucleus: 0E0\n')
        txtfile.write('NameOfPatient: \n')
        txtfile.write('DateOfExperiment: \n')
        txtfile.write('Spectrometer: \n')
        txtfile.write('AdditionalInfo: \n')
        txtfile.write('SignalNames:\n')
        txtfile.write('\n\n')
        txtfile.write('Signal and FFT\n')
        txtfile.write('sig(real)	sig(imag)\n')
        if df == 1:
            txtfile.write(f'Signal 1 out of {df} in file\n')
            for t in range(0,pd):
                txtfile.write(f'{re[t]}\t{imag[t]}\n')
        else:
            for idx in range(0,df):
                txtfile.write(f'Signal {idx+1} out of {df} in file\n')
                for t in range(0,pd):
                    txtfile.write(f'{re[t][idx]}\t{imag[t][idx]}\n')
                txtfile.write('\n')
        txtfile.close()

def makeHeader(fn,pd,df,t,zp,bt,tf):
    header = {}
    header['fn'] = fn
    header['PointsInDataset']= pd
    header['DatasetsInFile'] = df
    header['t'] = t
    header['ZeroOrderPhase'] = zp
    header['BeginTime']= bt
    header['Transfreq']= tf
    return header
