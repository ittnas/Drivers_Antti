#!/usr/bin/env python

from PyDAQmx.DAQmxFunctions import *
from PyDAQmx.DAQmxConstants import *
import PyDAQmx


import InstrumentDriver
import numpy as np

class Error(Exception):
    pass

class Driver(InstrumentDriver.InstrumentWorker):
    """ This class implements the NI DAQ card"""

    def performOpen(self, options={}):
        """Perform the operation of opening the instrument connection"""
        # init object variables
        com_id = int(self.comCfg.address)
        self.com_id = com_id
        self.nCh = 16
        self.lTrace = [np.array([])]*self.nCh
        self.lChName = [(b"Dev%d/ai%d" % (com_id, n)) for n in range(self.nCh)]
        self.lChDig = [(b"Dev%d/port0/line%d" % (com_id, n)) for n in range(self.nCh)]
        self.lVoltName = [('Ch%d: Voltage' % (n+1)) for n in range(self.nCh)]
        self.lSignalName = [('Ch%d: Data' % (n+1)) for n in range(self.nCh)]
        self.dt = 1.0
        self.mAI = None
        # configure DAQ 
        self.configureDAQ()


    def configureDAQ(self):
        """Configure DAQ according to settings"""
        # close old daq object, if in use
        if self.mAI is not None:
            self.mAI.closeAll()
        # get config
        nSample = int(self.getValue('Number of samples'))
        rate = float(self.getValue('Sample rate'))
        trigSource = self.getValue('Trig source')
        # check trig type
        if trigSource == 'Immediate':
            trigSource = None
            trigSlopePositive = True
            trigLevel = 0.0
        else:
            # trig from channel
            iTrig = self.getValueIndex('Trig source') - 1 # int(trigSource[-1])
            if iTrig < len(self.lChName):
                trigSource = self.lChName[iTrig]
            else:
                trigSource = self.lChDig[iTrig-len(self.lChName)]
            trigSlopePositive = (self.getValue('Trig slope')=='Positive')
            trigLevel = float(self.getValue('Trig level'))
        # get channels and limits
        limit = []
        lCh = []
        for n in range(self.nCh):
            s = 'Ch%d: ' % (n+1)
            # check if enabled
            if self.getValue(s+'Enabled'):
                limit.append([self.getValue(s+'Low range'),
                              self.getValue(s+'High range')])
                lCh.append(self.lChName[n])
        # open connection
        self.mAI = MultiChannelAnalogInput(lCh, limit, com_id=self.com_id)
        self.mAI.configure(rate, nSample, trigSource, trigSlopePositive, trigLevel)


    def performClose(self, bError=False, options={}):
        """Perform the close instrument connection operation"""
        # check if digitizer object exists
        try:
            if self.mAI is None:
                # do nothing, object doesn't exist (probably was never opened)
                return
        except:
            # never return error here, do nothing, object doesn't exist
            return
        try:
            # close and remove object
            self.mAI.closeAll()
        except:
            # never return error here
            pass


    def performSetValue(self, quant, value, sweepRate=0.0, options={}):
        """Perform the Set Value instrument operation. This function should
        return the actual value set by the instrument"""
        # check if trig config is updated
        if self.isFirstCall(options):
            self.bNewConfig = False
        # start with setting current quant value
        quant.setValue(value)
        if quant.name.startswith('Output'):
            # set output
            ch = int(quant.name[7:])
            address = b"Dev%d/ao%d" % (self.com_id, (ch-1))
            # task
            task = PyDAQmx.Task()
            task.CreateAOVoltageChan(address,b"",-10.0,10.0,PyDAQmx.DAQmx_Val_Volts,None)
            task.StartTask()
            task.WriteAnalogScalarF64(1,10.0,value,None)
            task.StopTask()
        else:
            # for all other quantities, mark config as updated, to set at end
            self.bNewConfig = True
        # if final call, configure the DAQ based on the new settings
        if self.bNewConfig and self.isFinalCall(options):
            self.configureDAQ()
        return value




    def performGetValue(self, quant, options={}):
        """Perform the Get Value instrument operation"""
        # only perform instrument operation for data calls
        if quant.name in (self.lSignalName + self.lVoltName):
            # get index
            if quant.name in self.lSignalName:
                bSignal = True
                indx = self.lSignalName.index(quant.name)
            else:
                bSignal = False
                indx = self.lVoltName.index(quant.name)
            # check if first call, if so get new traces
            if self.isFirstCall(options):
                self.getTraces()
            # return correct data
            if bSignal:
                # signal, return vector
                value = quant.getTraceDict(self.lTrace[indx], dt=self.dt)
            else:
                # voltage, return single value
                value = np.mean(self.lTrace[indx])
        else:
            # just return the quantity value
            value = quant.getValue()
        return value


    def getTraces(self):
        """Resample the data"""
        data = self.mAI.readAll()
        # put data in list of channels
        for key, data in data.items():
            indx = self.lChName.index(key)
            self.lTrace[indx] = data
        self.dt = 1.0/self.getValue('Sample rate')



class MultiChannelAnalogInput(object):
    """Class to create a multi-channel analog input
    
    Usage: AI = MultiChannelInput(physicalChannel)
        physicalChannel: a string or a list of strings
    optional parameter: limit: tuple or list of tuples, the AI limit values
                        reset: Boolean
    Methods:
        read(name), return the value of the input name
        readAll(), return a dictionary name:value
    """
    def __init__(self,physicalChannel, limit = None, reset = False, com_id=1):
        self.taskHandle = None
        self.com_id = com_id
        if type(physicalChannel) == type(b""):
            self.physicalChannel = [physicalChannel]
        else:
            self.physicalChannel  =physicalChannel
        self.nCh = physicalChannel.__len__()
        if limit is None:
            self.limit = dict([(name, (-10.0,10.0)) for name in self.physicalChannel])
        elif type(limit) == tuple:
            self.limit = dict([(name, limit) for name in self.physicalChannel])
        else:
            self.limit = dict([(name, limit[i]) for  i,name in enumerate(self.physicalChannel)])           
        if reset:
            DAQmxResetDevice(physicalChannel[0].split(b'/')[0] )
    def configure(self, rate=10000., nSample=1E3, trig=None, trigSlopePositive=True, trigLevel=0.0):
        # Create one task handle for all
        self.rate = rate
        self.nSample = nSample
        self.taskHandle = TaskHandle(0)
        DAQmxCreateTask(b"",byref(self.taskHandle))
        for name in self.physicalChannel:
            DAQmxCreateAIVoltageChan(self.taskHandle,name,b"",DAQmx_Val_RSE,
                                     self.limit[name][0],self.limit[name][1],
                                     DAQmx_Val_Volts,None)
        DAQmxCfgSampClkTiming(self.taskHandle,b"",float(rate),DAQmx_Val_Rising,DAQmx_Val_FiniteSamps,int(nSample))
        if trig is None:
            DAQmxDisableStartTrig(self.taskHandle)
        elif str(trig).startswith('Dev%d/port' % self.com_id):
            # digital trigger
            trigEdge = DAQmx_Val_Rising if trigSlopePositive else DAQmx_Val_Falling 
            DAQmxCfgDigEdgeStartTrig(self.taskHandle,str(trig),trigEdge)
        else:
            # analog trigger
            trigSlope = DAQmx_Val_RisingSlope if trigSlopePositive else DAQmx_Val_FallingSlope 
            DAQmxCfgAnlgEdgeStartTrig(self.taskHandle,str(trig),trigSlope,trigLevel)
    def readAll(self):
        DAQmxStartTask(self.taskHandle)
        data = np.zeros((self.nCh*self.nSample,), dtype=np.float64)
#        data = AI_data_type()
        read = int32()
        DAQmxReadAnalogF64(self.taskHandle,self.nSample,10.0,DAQmx_Val_GroupByChannel,data,len(data),byref(read),None)
        DAQmxStopTask(self.taskHandle)
        # output data as dict
        dOut = dict()
        for n, name in enumerate(self.physicalChannel):
            dOut[name] = data[(n*self.nSample):((n+1)*self.nSample)]
        return dOut
    def closeAll(self):
        # close all channels
        if self.taskHandle is None:
            return
        DAQmxStopTask(self.taskHandle)
        DAQmxClearTask(self.taskHandle)
        


if __name__ == '__main__':
    pass
