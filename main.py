import mapping
import trainingLibrary
import model

# set .mid and .midi to windows media player

# ---------------------------------------------------

# trainingLibrary.playRandomSelectionFromCorpus()

# trainingLibrary.playRandomSelectionFromPersonalCollection() # x random

# ---------------------------------------------------

trainingLibrary.getDataFromRandomSelectionOfSongsFromCorpus()  # 101 random

trainingLibrary.getDataFromAllSongsInFolder()  # 101 songs

# ---------------------------------------------------

mapping.mappingNoteToIntAllSongs()

model.LSTMFunction()

# trainingLibrary.train_network()

# trainingLibrary.generateOUTPUT()
