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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
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
    # print('modelInputV3')
    # print(modelInput)
    # print('len(modelInput)V3')
    # print(len(modelInput))
    numberOfSequences = len(modelInput)
    print('numberOfSequences')
    print(numberOfSequences)
    
    # reshape
    modelInput = numpy.reshape(modelInput, (numberOfSequences, sequenceLength, 1))
    # print('modelInputReshapedV3')
    # print(modelInput)
    # print('modelInput.shapeV3')
    # print(modelInput.shape)
    
    # normalize
    numberOfUniqueElements = len(numpy.unique(noteAndDurationToIntOneList))
    # print('numberOfUniqueElementsV3')
    # print(numberOfUniqueElements)
    modelInput = modelInput / float(numberOfUniqueElements)
    # print('modelInputReshapedAndNormalizedV3')
    # print(modelInput)
    print('modelInput.shapeV3')
    print(modelInput.shape)

    # print('modelTargetV3')
    # print(modelTarget)
    modelTargetUpdated = []
    for data in modelTarget:
        data = data[0]
        modelTargetUpdated.append(data)
    # print('modelTargetUpdatedV3')
    # print(modelTargetUpdated)
    modelTarget = np_utils.to_categorical(modelTargetUpdated)
    # print('modelTargetV3')
    # print(modelTarget)
    print('modelTarget.shapeV3')
    print(modelTarget.shape)

    # TRAIN MODEL

    inputShape = (sequenceLength, 1)
    print('inputShapeV3')
    print(inputShape)

    model = Sequential()
    model.add(CuDNNLSTM(
        128,
        input_shape=inputShape,
        return_sequences=True
    ))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    filepath = 'model-best-.hdf5'
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
    )
    callbacks_list = [checkpoint]

    model.fit(x=modelInput,
              y=modelTarget,
              epochs=10,
              verbose=1,
              callbacks=callbacks_list,
              )

    # output  --------------------------------------------------------------------------------------
    # USE TRAINED MODEL FOR OUTPUT
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
    # print('modelInputNormalizedV3')
    # print(modelInputNormalized)
    # print('modelInputNormalizedV3.shape')
    # print(modelInputNormalized.shape)
    outputUpdated = []
    for data in output:
        data = data[0]
        outputUpdated.append(data)
    # print('outputV3')
    # print(outputUpdated)

    # OUTPUT MODEL

    inputShape = (sequenceLength, 1)
    print('inputShapeV3')
    print(inputShape)

    model = Sequential()
    model.add(CuDNNLSTM(
        128,
        input_shape=inputShape,
        return_sequences=True
    ))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.load_weights('model-best-.hdf5')

    print(model.summary())

    # get prediction from model
    # start by getting 100 sequence from model input
    startInt = numpy.random.randint(0, len(modelInputForOutput)-1)
    # print('startIntV3')
    # print(startInt)
    sequence = modelInputForOutput[startInt]
    # print('sequenceV3')
    # print(sequence)
    # print('len(sequenceFromStartInt)V3')
    # print(len(sequenceFromStartInt))

    predictedSong = []
    for i in range(10):
        # reshape sequence to 3D
        predictionInputReshaped = numpy.reshape(sequence, (1, len(sequence), 1))
        # print('predictionInputReshapedV3')
        # print(predictionInputReshaped)

        # normalize sequence
        predictionInputNormalized = predictionInputReshaped / float(numberOfUniqueElements)
        # print('predictionInputNormalizedV3')
        # print(predictionInputNormalized)

        # get prediction
        prediction = model.predict(predictionInputNormalized, verbose=0)
        # print('PREDICTION ---- PREDICTION OUTPUT - V3')
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
        # print('indexWithMostPredictionsV3')
        # print(indexWithMostPredictions)

        # convert int back to note and duration and append to list
        notePitchAndDurationMatch = noteAndDurationToIntWithEnumerate[indexWithMostPredictions]
        notePitchAndDurationMatch = notePitchAndDurationMatch[1]
        # print('notePitchAndDurationMatchV3')
        # print(notePitchAndDurationMatch)

        # append to predicted song list for later midi conversion
        predictedSong.append(notePitchAndDurationMatch)
        # print('predictedSongV3')
        # print(predictedSong)

        sequence.append(indexWithMostPredictions)
        # print('sequenceWithNewIndexAppendedV3')
        # print(sequence)

        # remove first element from sequence to create updated sequence for next prediction
        sequence = sequence[1:len(sequence)]
        # print('sequenceWithFirstElementRemovedV3')
        # print(sequence)
        # print('\n')
    print('predictedSongV3')
    print(predictedSong)

    # convert note:duration to music 21 format for midi output
    outputComposition = []
    for element in predictedSong:
        noteToSet = element.split(':')[0]
        durationToSet = element.split(':')[1]
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
        # print('outputCompositionV3')
        # print(outputComposition)
        # print('len(outputComposition)V3')
        # print(len(outputComposition))
    # print('outputCompositionV3')
    # print(outputComposition)
    midiStream = stream.Stream(outputComposition)
    midiStream.write('midi', fp='-Midi_Output/testOutputATTEMPT3.mid')
    midiStream.show('midi')
