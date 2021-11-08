# import tensorflow.keras.preprocessing
# from keras.preprocessing.sequence import TimeseriesGenerator
# from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import LSTM, CuDNNLSTM, Dense, Dropout, Activation, BatchNormalization, TimeDistributed, Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

import logging
import os
import numpy
import mapping

from music21 import *

# used to disable tensorflow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').disabled = True

def LSTMFunction():
    # input --------------------------------------------------------------------------------------
    # get mapping data - list of notes
    noteAndDurationToIntMultipleLists = mapping.noteAndDurationToIntMultipleLists
    # print('noteAndDurationToIntMultipleLists')
    # print(noteAndDurationToIntMultipleLists)

    numberOfSongs = (len(noteAndDurationToIntMultipleLists))
    # print('\n')
    print('numberOfSongs')
    print(numberOfSongs)

    # extend data for model input
    # extended by looping smaller songs to the same length of the longest song
    largestNumberOfElements = 0
    for song in noteAndDurationToIntMultipleLists:
        # print('\n')
        # print('Total Number Of Elements:')
        # print(len(song))
        if len(song) > largestNumberOfElements:
            largestNumberOfElements = len(song)
    # print('\n')
    print('largestNumberOfElements')
    print(largestNumberOfElements)
    for song in noteAndDurationToIntMultipleLists:
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
    # print('noteAndDurationToIntMultipleLists')
    # print(noteAndDurationToIntMultipleLists)
    noteAndDurationToIntOneList = mapping.noteAndDurationToIntOneList
    # print('noteAndDurationToIntOneList')
    # print(noteAndDurationToIntOneList)
    noteAndDurationToIntWithEnumerate = mapping.noteAndDurationToIntWithEnumerate
    # print('noteAndDurationToIntWithEnumerate')
    # print(noteAndDurationToIntWithEnumerate)

    sequenceLength = 100
    modelInput = []
    modelTarget = []
    for i in range(0, len(noteAndDurationToIntOneList) - sequenceLength, 1):
        sequenceIn = noteAndDurationToIntOneList[i:i + sequenceLength]
        sequenceOut = noteAndDurationToIntOneList[i + sequenceLength]
        modelInput.append(sequenceIn)
        modelTarget.append(noteAndDurationToIntWithEnumerate[sequenceOut])
    # print('\n')
    # print('modelInput')
    # print(modelInput)
    # print('len(modelInput)')
    # print(len(modelInput))
    numberOfSequences = len(modelInput)
    print('numberOfSequences')
    print(numberOfSequences)
    # Reshape
    modelInput = numpy.reshape(modelInput, (numberOfSequences, sequenceLength, 1))
    # print('modelInputReshaped')
    # print(modelInput)
    # print('modelInput.shape')
    # print(modelInput.shape)
    # Normalize
    numberOfUniqueElements = len(numpy.unique(noteAndDurationToIntOneList))
    # print('numberOfUniqueElements')
    # print(numberOfUniqueElements)
    modelInput = modelInput / float(numberOfUniqueElements)
    # print('modelInputReshapedAndNormalized')
    # print(modelInput)
    print('modelInput.shape')
    print(modelInput.shape)

    # print('modelTarget')
    # print(modelTarget)
    modelTargetUpdated = []
    for data in modelTarget:
        data = data[0]
        modelTargetUpdated.append(data)
    # print('modelTargetUpdated')
    # print(modelTargetUpdated)
    modelTarget = np_utils.to_categorical(modelTargetUpdated)
    # print('modelTarget')
    # print(modelTarget)
    print('modelTarget.shape')
    print(modelTarget.shape)

    # TRAIN MODEL

    inputShape = (sequenceLength, 1)
    print('inputShape')
    print(inputShape)

    model = Sequential()
    # model version 1
    model.add(CuDNNLSTM(
        256,
        input_shape=inputShape,
        return_sequences=True
    ))
    model.add(CuDNNLSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(numberOfUniqueElements))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # model version 2
    '''
    model.add(Bidirectional(CuDNNLSTM(
        128,
        input_shape=inputShape,
        return_sequences=True
    )))
    model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
    model.add(TimeDistributed(Dense(64, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    '''

    print(model.summary())

    filepath = 'model-best-.hdf5'
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(x=modelInput,
              y=modelTarget,
              epochs=20,
              verbose=1,
              callbacks=callbacks_list,
              )

    # output --------------------------------------------------------------------------------------
    # use model for output
    modelInputForOutput = []
    output = []
    for i in range(0, len(noteAndDurationToIntOneList) - sequenceLength, 1):
        sequenceIn = noteAndDurationToIntOneList[i:i + sequenceLength]
        sequenceOut = noteAndDurationToIntOneList[i + sequenceLength]
        modelInputForOutput.append(sequenceIn)
        output.append(noteAndDurationToIntWithEnumerate[sequenceOut])
    # print('modelInputForOutput')
    # print(modelInputForOutput)
    # numberOfSequences = len(modelInputForOutput)
    # modelInputNormalized = numpy.reshape(modelInputForOutput, (numberOfSequences, sequenceLength, 1))
    # modelInputNormalized = modelInputNormalized / float(numberOfUniqueElements)
    # print('modelInputNormalized')
    # print(modelInputNormalized)
    # print('modelInputNormalized.shape')
    # print(modelInputNormalized.shape)

    outputUpdated = []
    for data in output:
        data = data[0]
        outputUpdated.append(data)

    # print('output')
    # print(outputUpdated)

    # OUTPUT MODEL

    inputShape = (sequenceLength, 1)
    # print('inputShape')
    # print(inputShape)

    model = Sequential()
    # model version 1
    model.add(CuDNNLSTM(
        256,
        input_shape=inputShape,
        return_sequences=True
    ))
    model.add(CuDNNLSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(numberOfUniqueElements))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # model version 2
    '''
    model.add(Bidirectional(CuDNNLSTM(
        128,
        input_shape=inputShape,
        return_sequences=True
    )))
    model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
    model.add(TimeDistributed(Dense(64, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    '''

    model.load_weights('model-best-.hdf5')

    print(model.summary())

    # get prediction from model
    # start by getting 100 sequence from model input
    startInt = numpy.random.randint(0, len(modelInputForOutput)-1)
    # print('startInt')
    # print(startInt)
    sequence = modelInputForOutput[startInt]
    # print('sequence')
    # print(sequence)
    # print('len(sequenceFromStartInt)')
    # print(len(sequenceFromStartInt))

    predictedSong = []
    # set number of note:duration elements for output composition
    for i in range(200):
        # reshape sequence to 3D
        predictionInputReshaped = numpy.reshape(sequence, (1, len(sequence), 1))
        # print('predictionInputReshaped')
        # print(predictionInputReshaped)

        # normalize sequence
        predictionInputNormalized = predictionInputReshaped / float(numberOfUniqueElements)
        # print('predictionInputNormalized')
        # print(predictionInputNormalized)

        # get prediction
        prediction = model.predict(predictionInputNormalized, verbose=0)
        # print('PREDICTION ---- PREDICTION OUTPUT')
        # print(prediction)

        # denormalize prediction
        denormalizedPrediction = prediction * float(numberOfUniqueElements)
        # print('denormalizedPrediction')
        # print(denormalizedPrediction)

        # round to nearest int
        denormalizedPredictionRounded = (numpy.round(denormalizedPrediction, 0)).astype(int)
        # print('denormalizedPredictionRounded')
        # print(denormalizedPredictionRounded)

        # reshape
        denormalizedPredictionRoundedReshaped = numpy.reshape(denormalizedPredictionRounded, -1)
        # print('denormalizedPredictionRoundedReshaped')
        # print(denormalizedPredictionRoundedReshaped)
        # print('len(denormalizedPredictionRoundedReshaped)')
        # print(len(denormalizedPredictionRoundedReshaped))
        # print('numberOfUniqueElements')
        # print(numberOfUniqueElements)

        # get index with highest frequency and its location from array
        # this target output array represents numberOfUniqueElements
        highestInt = 0
        for element in denormalizedPredictionRoundedReshaped:
            if element > highestInt:
                highestInt = element
        # index = numpy.where(denormalizedPredictionRoundedReshaped == highestInt)[0]
        # print('highestIntAndIndex')
        # print(highestInt, index)
        indexWithMostPredictions = int(numpy.argmax(prediction))
        # print('indexWithMostPredictions')
        # print(indexWithMostPredictions)

        # convert int back to note and duration and append to list
        notePitchAndDurationMatch = noteAndDurationToIntWithEnumerate[indexWithMostPredictions]
        notePitchAndDurationMatch = notePitchAndDurationMatch[1]
        # print('notePitchAndDurationMatch')
        # print(notePitchAndDurationMatch)

        # append to predicted song list for later midi conversion
        predictedSong.append(notePitchAndDurationMatch)
        # print('predictedSong')
        # print(predictedSong)

        sequence.append(indexWithMostPredictions)
        # print('sequenceWithNewIndexAppended')
        # print(sequence)

        # remove first element from sequence to create new sequence for next prediction
        sequence = sequence[1:len(sequence)]
        # print('sequenceWithFirstElementRemoved')
        # print(sequence)
        # print('\n')
    print('predictedSong')
    print(predictedSong)

    # create output composition
    outputComposition = []
    for element in predictedSong:
        noteToSet = element.split(':')[0]
        durationToSet = element.split(':')[1]
        # convert duration to float if in fraction format ex. '17/3'
        if '/' in durationToSet:
            durationToSet = durationToSet.split('/')
            num = int(durationToSet[0])
            denom = int(durationToSet[1])
            durationToSet = "{:.2f}".format(float(num/denom))
        # regular note
        if len(noteToSet) <= 3:
            newNote = note.Note(noteToSet)
            newNote.quarterLength = float(durationToSet)
            newNote.storedInstrument = instrument.Piano()
            outputComposition.append(newNote)
        # rest
        elif 'rest' in element:
            newRest = note.Rest(element)
            newRest.quarterLength = float(durationToSet)
            newRest.storedInstrument = instrument.Piano()
            outputComposition.append(newRest)
        # chord
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
        # print('outputComposition')
        # print(outputComposition)
        # print('len(outputComposition)')
        # print(len(outputComposition))
    # print('outputComposition')
    # print(outputComposition)
    midiStream = stream.Stream(outputComposition)
    midiStream.write('midi', fp='-outputComposition.mid')
    midiStream.show('midi')
