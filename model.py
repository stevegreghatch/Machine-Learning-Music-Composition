import collections
import copy
import random

import keras.layers
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization
from keras.layers import CuDNNLSTM
from sklearn.model_selection import train_test_split
from keras import utils
from keras.layers import TimeDistributed, Bidirectional

import logging
import os
import numpy
from music21 import note, chord, instrument, stream
from music21 import *

import mapping

# list init
modelTargetShifted = []
modelInputNormalized = []
modelTargetNormalized = []

# used to disable tensorflow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
logging.getLogger('tensorflow').disabled = True


def LSTMFunction():

    # input --------------------------------------------------------------------------------------
    # get mapping data
    # list of lists of data
    modelInput = mapping.noteToIntLists
    # print('Model Input:')
    # print(modelInput)

    numberOfSongs = (len(modelInput))
    print('\n')
    print('Total Number Of Songs:')
    print(numberOfSongs)

    print('\n')
    print('Reshaping data for model input')

    # reshape data for model input
    largestNumberOfElements = 0
    for song in modelInput:
        # print('\n')
        # print('Total Number Of Elements:')
        # print(len(song))

        if len(song) > largestNumberOfElements:
            largestNumberOfElements = len(song)
    print('\n')
    print('largestNumberOfElements:')
    print(largestNumberOfElements)

    # make all song lengths the same to ensure same shape for model input
    # extended by looping smaller songs to the same length of the longest song
    for song in modelInput:
        for element in song:
            if len(song) < largestNumberOfElements:
                song.append(element)
        # print('\n')
        # print('New Total Number Of Elements:')
        # print(len(song))
        # print('\n')
        # print('Extended Song')
        # print(song)
    # print('\n')
    # print('New Model Input:')
    # print(modelInput)

    # reshape data to column array
    arrayInput = numpy.array(modelInput)
    reshapedArrayInput = numpy.column_stack(arrayInput)
    # print('\n')
    # print('reshapedArrayInput:')
    # print(reshapedArrayInput)

    modelInputReshaped = numpy.reshape(reshapedArrayInput, (1, largestNumberOfElements, numberOfSongs))
    # numberOfSongs, largestNumberOfElements, 1
    # print('\n')
    # print('Model Input Reshaped:')
    # print(modelInputReshaped)
    print('\n')
    print('modelInputReshaped.shape')
    print(modelInputReshaped.shape)

    # normalize data
    numberOfUniqueElements = len(numpy.unique(mapping.noteToIntList))
    # print('\n')
    # print('Number of Unique Elements:')
    # print(numberOfUniqueElements)

    for song in modelInputReshaped:
        normalizedList = song / float(numberOfUniqueElements)
        modelInputNormalized.append(normalizedList)
    # print('\n')
    # print('Model Input Reshaped and Normalized:')
    # print(modelInputNormalized)

    # target  --------------------------------------------------------------------------------------
    # copy same size int model input data
    modelTarget = copy.deepcopy(modelInput)
    # print('\n')
    # print('Model Target:')
    # print(modelTarget)

    # shift data - move all over by one to establish target sequence
    for song in modelTarget:
        shiftedList = collections.deque(song)
        shiftedList.rotate(-1)
        shiftedList = list(shiftedList)
        modelTargetShifted.append(shiftedList)
    # print('\n')
    # print('Model Target Shifted:')
    # print(modelTargetShifted)

    # reshape data to column array
    arrayOutput = numpy.array(modelTargetShifted)
    reshapedArrayTarget = numpy.column_stack(arrayOutput)
    # print('\n')
    # print('reshapedArrayTarget:')
    # print(reshapedArrayTarget)

    # reshape target data
    modelTargetReshaped = numpy.reshape(reshapedArrayTarget, (1, largestNumberOfElements, numberOfSongs))
    # print('\n')
    # print('Model Target Reshaped:')
    # print(modelTargetReshaped)
    print('modelTargetReshaped.shape')
    print(modelTargetReshaped.shape)

    # normalize output data
    for song in modelTargetReshaped:
        normalizedList = song / float(numberOfUniqueElements)
        modelTargetNormalized.append(normalizedList)
    # print('\n')
    # print('Model Target Reshaped and Normalized:')
    # print(modelTargetNormalized)

    # LSTM model -------------------------------------

    inputShape = largestNumberOfElements, numberOfSongs
    print('inputShape:')
    print(inputShape)

    model = Sequential()

    model.add(Bidirectional(CuDNNLSTM(numberOfSongs, input_shape=inputShape, return_sequences=True)))
    model.add(Bidirectional(CuDNNLSTM(numberOfSongs, input_shape=inputShape, return_sequences=True)))
    model.add(TimeDistributed(Dense(numberOfSongs, activation='sigmoid')))

    # maxing out around 0.3865 accuracy

    # model.add(CuDNNLSTM(numberOfSongs, input_shape=inputShape, return_sequences=True))
    # model.add(CuDNNLSTM(numberOfSongs, input_shape=inputShape, return_sequences=True))
    # model.add(Dense(numberOfSongs))
    # model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    filepath = 'model-best-.hdf5'
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(numpy.array(modelInputNormalized),
              numpy.array(modelTargetNormalized),
              epochs=2000,
              verbose=1,
              callbacks=callbacks_list,
              )

    print('\n')
    print(model.summary())

    # model output -------------------------------------------------------------

    # load weights from model
    model.load_weights(filepath)

    # get prediction from model to for sequence
    # prediction needs to be in format int that was reshaped and normalized and converted to numpy array
    predictedSong = []
    c = 0
    startingIntList = []
    highestFrequencyInt = 0
    predictionList = []
    # create 100 note composition
    for i in range(100):
        # first, get random starting int
        if c == 0:
            startingIntList = []
            startingInt = mapping.noteToIntList[random.randint(0, len(mapping.noteToIntList) - 1)]
            # print('\n')
            # print('startingInt:')
            # print(startingInt)
            # make startingInt into list that is same size as model shape
            while len(startingIntList) < numberOfSongs:
                startingIntList.append(startingInt)
            # print('\n')
            # print('startingIntList:')
            # print(startingIntList)
            predictionList.append(startingInt)
        # get previous prediction to predict new note in sequence based on previous prediction
        if c > 0:
            startingIntList = []
            startingInt = highestFrequencyInt
            # print(highestFrequencyInt)
            # make startingInt into list that is same size as model shape
            while len(startingIntList) < numberOfSongs:
                startingIntList.append(startingInt)
            # print('\n')
            # print('startingIntList:')
            # print(startingIntList)
            predictionList.append(startingInt)

        # reshape data to array
        arrayInt = numpy.array(startingIntList)
        reshapedArrayInt = numpy.column_stack(arrayInt)
        # print('\n')
        # print('reshapedArrayInt:')
        # print(reshapedArrayInt)

        # reshape data to 3D
        startingIntReshaped = numpy.reshape(reshapedArrayInt, (1, 1, numberOfSongs))
        # print('\n')
        # print('startingIntReshaped')
        # print(startingIntReshaped)

        # normalize data
        normalizedStartingInt = startingIntReshaped / float(numberOfUniqueElements)
        # print('\n')
        # print('normalizedStartingInt ---- PREDICTION INPUT')
        # print(normalizedStartingInt)

        # get prediction
        prediction = model.predict(normalizedStartingInt, verbose=0)
        # print('\n')
        # print('PREDICTION ---- PREDICTION OUTPUT')
        # print(prediction)

        # denormalize prediction
        denormalizedPrediction = prediction * float(numberOfUniqueElements)
        # print('\n')
        # print('denormalizedPrediction')
        # print(denormalizedPrediction)
        # round to nearest int
        denormalizedPredictionRounded = (numpy.round(denormalizedPrediction, 0)).astype(int)
        # print('\n')
        # print('denormalizedPredictionRounded')
        # print(denormalizedPredictionRounded)

        # reshape
        denormalizedPredictionRoundedReshaped = numpy.reshape(denormalizedPredictionRounded, -1)
        # print('\n')
        # print('denormalizedPredictionRoundedReshaped')
        # print(denormalizedPredictionRoundedReshaped)

        # get highest frequency int from array
        counts = numpy.bincount(denormalizedPredictionRoundedReshaped)
        highestFrequencyInt = numpy.argmax(counts)
        # print('\n')
        # print('highestFrequencyInt')
        # print(highestFrequencyInt)

        # convert int back to note and duration and append to list
        for element in mapping.noteToIntWithData:
            if highestFrequencyInt == element[0]:
                notePitchAndDurationMatch = element[1]
                # print('\n')
                # print('notePitchAndDurationMatch')
                # print(notePitchAndDurationMatch)
                predictedSong.append(notePitchAndDurationMatch)

        c += 1

    # print('\n')
    # print('mapping.noteToIntWithData')
    # print(mapping.noteToIntWithData)

    print('\n')
    print('predictionList')
    print(predictionList)

    print('\n')
    print('PREDICTED SONG ------------------------------------ ')
    print(predictedSong)

    # convert note:duration to music 21 format for midi output
    outputComposition = []
    midiStream = stream.Stream()
    print('\n')
    for element in predictedSong:
        noteToSet = element.split(':')[0]
        durationToSet = element.split(':')[1]
        # print(durationToSet)
        if '/' in durationToSet:
            durationToSet = durationToSet.split('/')
            num = int(durationToSet[0])
            denom = int(durationToSet[1])
            durationToSet = "{:.2f}".format(float(num/denom))
        # print('\n')
        # print('len(noteToSet)')
        # print(len(noteToSet))
        # note - needs to be converted from string to music 21 note
        if len(noteToSet) <= 3:
            newNote = note.Note(noteToSet)
            newNote.quarterLength = float(durationToSet)
            newNote.storedInstrument = instrument.Piano()
            outputComposition.append(newNote)
        # chord - needs to be converted from string to list to music 21 chord
        else:
            # print('noteToSet')
            # print(noteToSet)
            cToFiler = ["[", "'", "]", ","]
            for c in cToFiler:
                noteToSet = noteToSet.replace(c, '')
            # print('noteToSetUpdated')
            # print(noteToSet)
            noteToSetSplit = noteToSet.split()
            # print('noteToSetSplit')
            # print(noteToSetSplit)
            newChord = chord.Chord(noteToSetSplit)
            # print('newChord')
            # print(newChord)
            # print('newChord.pitches')
            # print(newChord.pitches)
            newChord.storedInstrument = instrument.Piano()
            newChord.quarterLength = float(durationToSet)
            outputComposition.append(newChord)
    print('\n')
    print('outputComposition')
    print(outputComposition)
    midiStream.append(outputComposition)
    midiStream.write('midi', fp='C:/Users/steve/Desktop/Midi_Output/testOutput.mid')
    midiStream.show('midi')
