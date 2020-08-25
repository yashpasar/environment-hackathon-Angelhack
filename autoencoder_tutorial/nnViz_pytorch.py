import matplotlib.pylab as plt
from math import cos, sin, atan
import numpy as np
from matplotlib.collections import PatchCollection, LineCollection

verticalDistanceBetweenLayers = 5.5
horizontalDistanceBetweenNeurons = 2
neuronRadius = 0.75
nNeuronsInWidestLayer = 4 

class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y      

    def draw(self):
        global neuronRadius, neuronPatches
        circle = plt.Circle((self.x, self.y), radius=neuronRadius, fill=True)
        circle.set_edgecolor((0,0,0))
        circle.set_facecolor((1,1,1))
        circle.set_linewidth(2)
        circle.set_zorder(2)
        
        neuronPatches.append(circle)

class Layer():
    def __init__(self, network, nNeurons, weights):
        self.prevLayer = self.get_prevLayer(network)
        self.y = self.compute_layer_vertical_pos()
        self.neurons = self.initialize_neurons(nNeurons)
        self.weights = weights
        self.maxWeight = 5
        self.minWeight = .25

    def initialize_neurons(self, nNeurons):
        global horizontalDistanceBetweenNeurons
        neurons = []
        x = self.compute_left_margin(nNeurons)
        for iteration in range(nNeurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += horizontalDistanceBetweenNeurons
        return neurons

    def compute_left_margin(self, nNeurons):
        global horizontalDistanceBetweenNeurons
        global nNeuronsInWidestLayer
        return horizontalDistanceBetweenNeurons * (nNeuronsInWidestLayer - nNeurons) / 2

    def compute_layer_vertical_pos(self):
        global verticalDistanceBetweenLayers
        if self.prevLayer:
            return self.prevLayer.y + verticalDistanceBetweenLayers
        else:
            return 0

    def get_prevLayer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def draw_edge(self, neuron1, neuron2, linewidth, sign):
        global neuronRadius, edgePatches, lineWidths, lineColors
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        xOffset = neuronRadius * sin(angle)
        yOffset = neuronRadius * cos(angle)
        lineXTuple = (neuron1.x - xOffset, neuron2.x + xOffset)
        lineYTuple = (neuron1.y - yOffset, neuron2.y + yOffset)

        if sign > 0:
            lineColor = (0, 0, .5, .75) # positive weights in red, 75% transparency
        else:
            lineColor = (.5, 0, 0, .75) # negative weights in blue, 75% transparency
       
        edgePatches.append([ ( lineXTuple[0], lineYTuple[0] ), ( lineXTuple[1], lineYTuple[1] ) ])
        lineWidths.append(linewidth)
        lineColors.append(lineColor)

    def draw(self):
        for iNeuronCurrentLayer in range(len(self.neurons)):
            neuron = self.neurons[iNeuronCurrentLayer]

            if self.prevLayer:
                for iNeuronPreviousLayer in range(len(self.prevLayer.neurons)):
                    prevLayer_neuron = self.prevLayer.neurons[iNeuronPreviousLayer]

                    if self.prevLayer.weights is not None:
                        
                        rawWeight = self.prevLayer.weights[iNeuronCurrentLayer, iNeuronPreviousLayer]
                        
                        if rawWeight > 0:
                            sign = 1
                        else:
                            sign = -1

                        processedWeight = self.minWeight + abs( self.prevLayer.weights[iNeuronCurrentLayer, iNeuronPreviousLayer])
                        
                        weight = min ( (self.maxWeight , max( self.minWeight,  processedWeight)) )
                        
                        #print ( str(weight) + ', ' + str( rawWeight ))
                    else:
                        weight = self.minWeight
                    self.draw_edge(neuron, prevLayer_neuron, weight, sign)
            neuron.draw()

class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, nNeurons, weights=None):
        layer = Layer(self, nNeurons, weights)
        self.layers.append(layer)

    def draw(self, ax, edgeSkip):    
        global neuronPatches, edgePatches, lineWidths, lineColors
        
        for layer in self.layers:
            layer.draw()
        
        edgeSkip = len(edgePatches)//10000
        print(edgeSkip)

        neuronCollection = PatchCollection(neuronPatches, match_original=True)
        print(len(edgePatches))
        edgeCollection = LineCollection(edgePatches[::edgeSkip], linewidths = lineWidths[::edgeSkip], colors=lineColors[::edgeSkip], zorder=0)        

        ax.add_collection(neuronCollection)
        ax.add_collection(edgeCollection)
        
neuronPatches = []
edgePatches = []
lineWidths = []
lineColors = []

def visualize_model( model, ax, edgeSkip = 1 ):    
    global neuronPatches, edgePatches, lineWidths, lineColors
    neuronPatches = []
    edgePatches = []
    lineWidths = []
    lineColors = []

    modelWeights = {}
    modelShape = {}

    iLayer = 0
    for i, iModule in enumerate(model.modules()):
        if hasattr(iModule, 'in_features'):
            modelShape[iLayer] = [ iModule.in_features, iModule.out_features ]
            modelWeights[iLayer] = list(iModule.parameters())[0].data.cpu().numpy()
            iLayer += 1
        
    nLayers = iLayer
    
    network = NeuralNetwork()
   
    # first layer
    network.add_layer( modelShape[0][0], modelWeights[0])

    for iLayer in range( nLayers - 1):    
        network.add_layer( modelShape[iLayer][1], modelWeights[iLayer+1] )

    # last layer
    network.add_layer(modelShape[nLayers-1][1])

    network.draw(ax, edgeSkip)