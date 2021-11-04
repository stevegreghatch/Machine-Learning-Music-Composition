import copy
from music21 import *
import random
import os
import logging
import numpy
import mapping

import collections

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
from keras.utils import np_utils
from keras.layers import TimeDistributed, Bidirectional

listOfNotesAndDurations = []
listOfAllNotesAndDurations = []


def getListOfNotesAndDurations(work):
    listOfNotesAndDurations.clear()
    noteToAdd = None
    notesToParse = work.flat.notes
    for element in notesToParse:
        if isinstance(element, note.Note):
            noteToAdd = str(element.pitch)
        elif isinstance(element, chord.Chord):
            noteToAdd = str([n.nameWithOctave for n in element.pitches])
        # time relative to quarter note = 1.0
        durationOfNote = str(element.quarterLength)
        listOfNotesAndDurations.append(noteToAdd + ':' + durationOfNote)
    # print(listOfNotesAndDurations)
    return listOfNotesAndDurations


# training set 1 -------------------------------------------------------------------------------------------------------
# corpus files from music21 - classical composers

listOfComposersInCorpus = ['bach', 'beach', 'beethoven', 'chopin', 'ciconia', 'corelli', 'cpebach',
                           'handel', 'haydn', 'joplin', 'josquin', 'monteverdi', 'mozart',
                           'palestrina', 'schoenberg', 'schubert', 'schumann', 'schumann_clara',
                           'verdi', 'weber']


# function for all data -----------------------------------------------
def getDataFromRandomSelectionOfSongsFromCorpus():
    listOfPathsOfFilesByComposer = []
    listOfPathsToChooseFrom = []
    listOfChosenPaths = []

    # get paths for all files by selected composers
    for composer in listOfComposersInCorpus:
        pathsOfFilesByComposer = corpus.getComposer(composer)
        listOfPathsOfFilesByComposer.append(pathsOfFilesByComposer)
    for filePaths in listOfPathsOfFilesByComposer:
        for filePath in filePaths:
            listOfPathsToChooseFrom.append(filePath)

    # selects desired number of random paths from selected composers from Corpus
    for i in range(400):
        chosenPath = random.choice(listOfPathsToChooseFrom)
        if chosenPath not in listOfChosenPaths:
            listOfChosenPaths.append(chosenPath)

    # get data from each path
    for path in listOfChosenPaths:
        work = converter.parse(path)
        # print('\n')
        # work.show('text')
        title = os.path.basename(os.path.dirname(path)).capitalize() + ' - ' + os.path.basename(path)
        data = getListOfNotesAndDurations(work)
        print(title)
        # print(data)
        # print('\n')
        listOfAllNotesAndDurations.append(copy.deepcopy(data))
    # print(listOfAllNotesAndDurations)


# training set 2 -------------------------------------------------------------------------------------------------------
# personal collection

# functions for random selection -------------------------------
def returnRandomPathFromPersonalCollection():
    folder = 'C:/Users/steve/Desktop/Midi_Files/'
    listOfFiles = os.listdir(folder)
    listOfFilePaths = []
    for file in listOfFiles:
        filePath = folder + file
        listOfFilePaths.append(filePath)
    selectedPath = random.choice(listOfFilePaths)
    return selectedPath


def playRandomSelectionFromPersonalCollection():
    for i in range(2):
        selectedPath = str(returnRandomPathFromPersonalCollection())
        work = converter.parse(selectedPath)
        title = os.path.basename(os.path.normpath(selectedPath))
        data = getListOfNotesAndDurations(work)
        print(title)
        # print(data)
        # print('\n')
        # work.show('midi')
        # work.show('text')
        listOfAllNotesAndDurations.append(copy.deepcopy(data))


# function for all data -----------------------------------------------
def getDataFromAllSongsInFolder():
    listOfFilePaths = []
    folder = 'C:/Users/steve/Desktop/Midi_Files/'
    listOfFiles = os.listdir(folder)

    # get list of file paths
    for file in listOfFiles:
        filePath = folder + file
        listOfFilePaths.append(filePath)

    # get data from each path
    for path in listOfFilePaths:
        work = converter.parse(path)
        title = os.path.basename(path)
        data = getListOfNotesAndDurations(work)
        print(title)
        # print(data)
        # print('\n')
        listOfAllNotesAndDurations.append(copy.deepcopy(data))
    # print(listOfAllNotesAndDurations)


# --------------------------------------------------------------------------------

# TRAIL NUMBER 2

def train_network():
    notes = get_notes()
    n_vocab = len(set(notes))
    network_input, network_output = prepare_sequences(notes, n_vocab)
    model = create_network(network_input, n_vocab)
    train(model, network_input, network_output)

# listOfAllNotesAndDurationsATTEMPT2 = []
LISTOFALLNOTES = []
def get_notes():
    notes = []
    listOfFilePaths = []
    folder = 'C:/Users/steve/Desktop/Midi_Files/'
    listOfFiles = os.listdir(folder)
    # get list of file paths
    for file in listOfFiles:
        filePath = folder + file
        listOfFilePaths.append(filePath)
    # get data from each path
    for path in listOfFilePaths:
        work = converter.parse(path)
        title = os.path.basename(path)
        print(title)
        notesToParse = work.flat.notesAndRests
        for element in notesToParse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
            elif isinstance(element, note.Rest):
                notes.append(element.name)
        # listOfAllNotesAndDurationsATTEMPT2.append(copy.deepcopy(notes))
    LISTOFALLNOTES.append(notes)
    return notes

def prepare_sequences(notes, n_vocab):
    sequence_length = 100
    # get all pitch names
    pitchNames = sorted(set(item for item in notes))
    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchNames))
    network_input = []
    network_output = []
    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
    n_patterns = len(network_input)
    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)
    network_output = np_utils.to_categorical(network_output)
    return (network_input, network_output)

def create_network(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def train(model, network_input, network_output):
    filepath = "model-best-.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    model.fit(network_input, network_output, epochs=10, batch_size=64, callbacks=callbacks_list)

# ---------------

def generateOUTPUT():
    notes = LISTOFALLNOTES
    pitchNames = sorted(set(item for item in notes))
    n_vocab = len(set(notes))
    network_input, normalized_input = prepare_sequencesOUTPUT(notes, pitchNames, n_vocab)
    model = create_networkOUTPUT(normalized_input, n_vocab)
    prediction_output = generate_notes(model, network_input, pitchNames, n_vocab)
    create_midi(prediction_output)

def prepare_sequencesOUTPUT(notes, pitchNames, n_vocab):
    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchNames))
    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])
    n_patterns = len(network_input)
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    normalized_input = normalized_input / float(n_vocab)
    return (network_input, normalized_input)

def create_networkOUTPUT(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.load_weights('model-best-.hdf5')
    return model

def generate_notes(model, network_input, pitchNames, n_vocab):
    start = numpy.random.randint(0, len(network_input)-1)
    int_to_note = dict((number, note) for number, note in enumerate(pitchNames))
    pattern = network_input[start]
    prediction_output = []
    for note_index in range(100):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)
        index = int(numpy.argmax(prediction))
        result = int_to_note[index]
        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    return prediction_output

def create_midi(prediction_output):
    offset = 0
    output_notes = []
    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        elif 'rest' in pattern:
            new_rest = note.Rest(pattern)
            new_rest.offset = offset
            new_rest.storedInstrument = instrument.Piano()
            output_notes.append(new_rest)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='C:/Users/steve/Desktop/Midi_Output/testOutputATTEMPT2.mid')
