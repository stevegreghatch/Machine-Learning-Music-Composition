from music21 import *

import model

from matplotlib import pyplot, image

from datetime import datetime

# Data Visualization -------------------------------------------------

def lossVsEpochGraph():
    # vis1 = 'Loss vs Epoch'
    pyplot.figure()
    loss = model.getHistoryLoss()
    pyplot.plot(loss)
    pyplot.title('Loss vs Epoch')
    pyplot.ylabel('Loss')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Loss'], loc='upper right')
    pyplot.savefig('C:/Users/steve/Desktop/Graph_Output/Loss vs Epoch.png')
    pyplot.close()

def accuracyVsEpochGraph():
    # vis2 = 'Accuracy vs Epoch'
    pyplot.figure()
    accuracy = model.getAccuracyHistory()
    pyplot.plot(accuracy)
    pyplot.title('Accuracy vs Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Accuracy'], loc='upper right')
    pyplot.savefig('C:/Users/steve/Desktop/Graph_Output/Accuracy vs Epoch.png')
    pyplot.close()

def noteQuarterLengthByPitch():
    # vis3 = 'Note Quarter Length by Pitch'
    midiStream = model.getMidiStream()
    g = graph.plot.HorizontalBarPitchSpaceOffset(midiStream)
    g.doneAction = None
    g.run()
    g.write(fp='C:/Users/steve/Desktop/Graph_Output/Note Quarter Length By Pitch.png')


def pitchByQuarterLengthScatterWeighted():
    # vis4 = 'Pitch by Quarter Length Scatter (weighted)'
    midiStream = model.getMidiStream()
    g = graph.plot.ScatterWeightedPitchSpaceQuarterLength(midiStream)
    g.doneAction = None
    g.run()
    g.write(fp='C:/Users/steve/Desktop/Graph_Output/Count of Pitch and Quarter Length.png')

def pitchHistogram():
    # vis5 = 'Pitch Histogram'
    midiStream = model.getMidiStream()
    g = graph.plot.HistogramPitchSpace(midiStream)
    g.doneAction = None
    g.run()
    g.write(fp='C:/Users/steve/Desktop/Graph_Output/Pitch Histogram.png')

def dash():

    # call functions to create files to import
    lossVsEpochGraph()
    accuracyVsEpochGraph()
    noteQuarterLengthByPitch()
    pitchByQuarterLengthScatterWeighted()
    pitchHistogram()

    currentTime = datetime.now().strftime('%H:%M %m-%d-%Y')

    fig = pyplot.figure('Dashboard ' + currentTime, facecolor='tab:cyan', figsize=(24.75, 12))

    ax1 = pyplot.subplot2grid((3, 3), (0, 0), rowspan=1, colspan=1, fig=fig)
    ax2 = pyplot.subplot2grid((3, 3), (0, 1), rowspan=1, colspan=1, fig=fig)
    ax3 = pyplot.subplot2grid((3, 3), (1, 0), rowspan=2, colspan=2, fig=fig)
    ax4 = pyplot.subplot2grid((3, 3), (0, 2), rowspan=1, colspan=1, fig=fig)
    ax5 = pyplot.subplot2grid((3, 3), (1, 2), rowspan=2, colspan=1, fig=fig)

    # vis1 = 'Loss vs Epoch'
    vis1IMG = pyplot.imread('C:/Users/steve/Desktop/Graph_Output/Loss vs Epoch.png')
    ax1.imshow(vis1IMG)
    ax1.axis('off')

    # vis2 = 'Accuracy vs Epoch'
    vis2IMG = pyplot.imread('C:/Users/steve/Desktop/Graph_Output/Accuracy vs Epoch.png')
    ax2.imshow(vis2IMG)
    ax2.axis('off')

    # vis3 = 'Note Quarter Length by Pitch'
    vis3IMG = image.imread('C:/Users/steve/Desktop/Graph_Output/Note Quarter Length By Pitch.png')
    ax3.imshow(vis3IMG)
    ax3.axis('off')

    # vis4 = 'Pitch Histogram'
    vis4IMG = image.imread('C:/Users/steve/Desktop/Graph_Output/Pitch Histogram.png')
    ax4.imshow(vis4IMG)
    ax4.axis('off')

    # vis5 = 'Pitch by Quarter Length Scatter (weighted)'
    vis5IMG = image.imread('C:/Users/steve/Desktop/Graph_Output/Count of Pitch and Quarter Length.png')
    ax5.imshow(vis5IMG)
    ax5.axis('off')

    fig.tight_layout()

    fig.subplots_adjust(wspace=0.0, left=0.0, right=0.95)

    pyplot.show()

    # pyplot.close(fig)

