import collections
import copy
import random
import logging
import os
import numpy
import mapping

from music21 import *

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, BatchNormalization, CuDNNLSTM, TimeDistributed, Bidirectional
from keras.callbacks import ModelCheckpoint
# from sklearn.model_selection import train_test_split
from keras import utils

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
    modelInputData = mapping.noteToIntLists
    # print('Model Input:')
    # print(modelInput)

    numberOfSongs = (len(modelInputData))
    print('\n')
    print('Total Number Of Songs:')
    print(numberOfSongs)

    print('\n')
    print('Reshaping data for model input')

    # reshape data for model input
    largestNumberOfElements = 0
    for song in modelInputData:
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
    for song in modelInputData:
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
    # print(modelInputData)

    # convert data to array
    arrayModelInputData = numpy.array(modelInputData)
    # print('\n')
    # print('arrayInput:')
    # print(arrayModelInputData)

    # normalize data
    numberOfUniqueElements = len(numpy.unique(mapping.noteToIntList))
    # print('\n')
    # print('Number of Unique Elements:')
    # print(numberOfUniqueElements)

    for data in arrayModelInputData:
        data = data / float(numberOfUniqueElements)
        modelInputDataNormalized.append(data)
    # print('\n')
    # print('modelInputDataNormalized')
    # print(modelInputDataNormalized)

    # reshape data
    modelInputDataNormalizedAndReshaped = numpy.reshape(modelInputDataNormalized,
                                                        (largestNumberOfElements*numberOfSongs, 1))

    # target  --------------------------------------------------------------------------------------
    # copy same size int model input data
    modelTargets = copy.deepcopy(modelInputData)
    # print('\n')
    # print('Model Target:')
    # print(modelTarget)

    # shift data - move all over by one to establish target sequence
    for song in modelTargets:
        shiftedList = collections.deque(song)
        shiftedList.rotate(-1)
        shiftedList = list(shiftedList)
        modelTargetsShifted.append(shiftedList)
    # print('\n')
    # print('Model Target Shifted:')
    # print(modelTargetShifted)

    # convert to array
    arrayModelTargetsShifted = numpy.array(modelTargetsShifted)
    # print('\n')
    # print('arrayModelTargetsShifted')
    # print(arrayModelTargetsShifted)

    # normalize output data
    for data in arrayModelTargetsShifted:
        data = data / float(numberOfUniqueElements)
        modelTargetDataNormalized.append(data)
    # print('\n')
    # print('Model Target Reshaped and Normalized:')
    # print(modelTargetDataNormalized)

    # reshape data
    modelTargetDataNormalizedAndReshaped = numpy.reshape(modelTargetDataNormalized,
                                                         (largestNumberOfElements*numberOfSongs, 1))
    print('\n')
    print('modelInputDataNormalizedAndReshaped.shape')
    print(modelInputDataNormalizedAndReshaped.shape)
    print('\n')
    print('modelTargetDataNormalizedAndReshaped.shape')
    print(modelTargetDataNormalizedAndReshaped.shape)

    # construct generator with both input data and target data
    generator = TimeseriesGenerator(modelInputDataNormalizedAndReshaped, modelTargetDataNormalizedAndReshaped,
                                    length=numberOfSongs)

    inputShape = (largestNumberOfElements*numberOfSongs, 1)
    print('\n')
    print('inputShape')
    print(inputShape)

    model = Sequential()
    model.add(CuDNNLSTM(
        512,
        input_shape=inputShape,
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(CuDNNLSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(CuDNNLSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(largestNumberOfElements))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    filepath = 'model-best-.hdf5'
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(generator,
              epochs=50,
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
    midiStream.write('midi', fp='-/Midi_Output/testOutput.mid')
    midiStream.show('midi')
