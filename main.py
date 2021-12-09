import mapping
import trainingLibrary
import model
import dashboard

# ---------------------------------------------------

trainingLibrary.getDataFromAllSongsInFolder()
trainingLibrary.getDataFromRandomSelectionOfSongsFromCorpus()
mapping.mappingNoteToIntAllSongs()
model.LSTMFunction()
dashboard.dash()
