# File containing TopObject class used for loading AnnData objects and performing core scTOP operations
# Author: Eitan Vilker (with some functions written by Maria Yampolskaya)

import numpy as np
import pandas as pd
import sctop as top
import scanpy as sc
import anndata as ad
from tqdm import tqdm
import inspect

class TopObject:
    def __init__(self, name, manualInit=False, useAverage=False, datasetInfo=None):
        self.name = name
        datasetInfo = getDatasetSpecificInfo(self.name)
        if len(datasetInfo) > 1:
            self.cellTypeColumn, self.toKeep, self.toExclude, self.filePath, self.timeColumn, self.duplicates, self.raw, self.layer = datasetInfo
        else:
            # Prompt to manually add dataset info
            pass
        self.projections = {}
        self.basis = None
        if not manualInit:  # In case you want to adjust any of the parameters first
            self.setAnnObject()

    # Summary of parameters upon printing object
    def __str__(self):
        attributeMap = inspect.getmembers(self)[2][1]
        keys = attributeMap.keys()
        toReturn = "Attributes:"
        for key in keys:
            value = attributeMap[key]
            valueType = type(value)
            if valueType is str or valueType is bool:
                toReturn += "\n" + key + ": " + str(value)
            elif valueType is list or valueType is np.ndarray:
                if len(value) > 0 and type(value[0]) is np.ndarray:
                    value = value[0]
                if len(value) > 5:
                    toReturn += "\n" + key + ": [" + ", ".join(list(value[:5])) + "]..."
                else:
                    toReturn += "\n" + key + ": " + str(value)
            elif valueType is pd.core.frame.DataFrame:
                toReturn += "\n" + key + ": " + str(list(value.columns)[:5]) + "..."
            elif valueType is pd.core.series.Series:
                toReturn += "\n" + key + ": " + str(list(value)[:5]) + "..."
            elif value is None:
                toReturn += "\n" + key + ": None"
            else:
                toReturn += "\n" + key + ": " + str(valueType)
        return toReturn

    # Initialize AnnData object
    def setAnnObject(self, useAverage=False):
        print("Setting AnnData object...")
        if not hasattr(self, "annObject"):
            self.annObject = sc.read_h5ad(self.filePath)
        if self.duplicates:
            self.annObject.var_names_make_unique()
        self.setMetadata()
        self.setDF()
        self.processDataset(useAverage=useAverage)

    # Set AnnData object to include or exclude cells with certain labels
    def filterAnnObject(self, toKeep=None, toExclude=None, keep=False, exclude=False):
        print("Filtering AnnData...")
        if keep:
            if toKeep is None:
                toKeep = self.toKeep
            self.annObject = self.annObject[self.annotations.isin(toKeep)]
        if exclude:
            if toExclude is None:
                toExclude = self.toExclude
            self.annObject = self.annObject[~self.annotations.isin(toExclude)]
        self.setMetadata()

    # Set key features. May need to be called whenever object is edited
    def setMetadata(self):
        print("Setting metadata...")
        self.metadata = self.annObject.obs
        self.annotations = self.metadata[self.cellTypeColumn]
        self.filteredAnnotations = [label if (label in self.toKeep and label not in self.toExclude) else "Other" for label in self.annotations]
        self.kwargs = {'s': 40,
                       'style': self.filteredAnnotations
        }
        self.timeSortFunction = None
        self.timesSorted = None
        if self.timeColumn is not None:
            self.timeSortFunction = lambda time: int("".join([char for char in time if char.isdigit()])) # if numbers in string unrelated to time this won't work
            self.timesSorted = sorted([str(time) for time in set(self.metadata[self.timeColumn])], key=self.timeSortFunction)

    # Set df, with a few extra options in case there are issues with the df
    def setDF(self, raw=False, duplicates=False, layer=None):
        print("Setting df...")
        if self.raw or raw:
            self.df = pd.DataFrame(self.annObject.raw.X.toarray(), index = self.metadata.index, columns = self.annObject.raw.var_names).T
        else:
            if layer is None:
                layer = self.layer
            self.df = self.annObject.to_df(layer=layer).T
        if self.duplicates or duplicates:
            self.df = self.df.drop_duplicates().groupby(level=0).mean()

    # First scTOP function, ranks and normalizes source 
    def processDataset(self, useAverage=False):
        print("Processing scTOP data...")
        self.processedData = top.process(self.df, average=useAverage)
        print("Done!")

    # Main scTOP function, computing similarity between labels in sources and basis
    def projectOntoBasis(self, basis, projectionName):
        print("Projecting onto basis...")
        projection = top.score(basis, self.processedData)
        self.projections[projectionName] = projection
        print("Finished projecting!")
        return projection

    # Using any dataset with well-defined clusters, set it as a basis
    def setBasis(self, useAllCells=True):
        print("Setting basis...")
        # Count the number of cells per type
        type_counts = self.annotations.value_counts()

        # Using fewer than 150-200 cells leads to nonsensical results, due to noise. More cells -> less sampling error
        threshold = 200 # only use cell types with at least this many cells (but use all cells for training)
        types_above_threshold = type_counts[type_counts > threshold].index
        basis_list = []
        training_IDs = []
        rng = np.random.default_rng()
        for cell_type in tqdm(types_above_threshold):
            cell_IDs = self.metadata[self.annotations == cell_type].index
            if not useAllCells:
                current_IDs = rng.choice(cell_IDs, size=threshold, replace=False) # This line is for using only the threshold number of cells for the reference basis. This can be useful for testing the accuracy of the basis, but it performs notably worse in accuracy metrics compared to using all possible cells.
            else:
                current_IDs = cell_IDs
            cell_data = self.df[current_IDs]
            training_IDs += [current_IDs] # Keep track of training_IDs so that you can exclude them if you want to test the accuracy

            # Average across the cells and process them using the scTOP processing method
            processed = top.process(cell_data, average=True)
            basis_list += [processed]

        training_IDs = np.concatenate(training_IDs)
        self.basis = pd.concat(basis_list, axis=1)
        self.basis.columns = types_above_threshold
        self.basis.index.name = "gene"
        test_IDs = np.setdiff1d(self.df.columns, training_IDs)
        # test_IDs = training_IDs
        self.split_IDs = np.array_split(test_IDs, 10) # From Maria- "I split this test dataset because it's very large and took up a lot of memory -- you don't need to do this if you have enough memory to test the entire dataset at once"
        print("Basis set!")
        return self.basis

    # Add the desired columns of a smaller basis to a primary basis
    def combineBases(self, otherBasis, colsToKeep, useAllCells=True, combinedBasisName="Combined"):
        print("Combining bases...")
        self.setBasis(useAllCells=useAllCells)
        if otherBasis.isinstance(TopObject):
            otherBasis.setBasis(useAllCells=useAllCells)
            basis2 = otherBasis.basis
        else:
            basis2 = otherBasis
        self.basis.index.name = basis2.index.name
        if not hasattr(self, "combinedBases"):
            self.combinedBases = {}
        combinedBasis = pd.merge(self.basis, basis2[colsToKeep], on=self.basis.index.name, how="inner")
        return self.combinedBases[combinedBasisName]

    # Test an existing basis (not combined). Optionally adjust the minimum accuracy threshold
    def testBasis(self, specification_value=0.1):
        print("Processing test data...")
        accuracies = {'top1': 0,
                      'top3': 0,
                      'unspecified': 0}
        matches = {}
        misses = {}
        for sample_IDs in tqdm(self.split_IDs):
            test_data = self.df[sample_IDs]
            test_processed = top.process(test_data)
            test_projections = top.score(self.basis, test_processed)
            accuracies, matches, misses = self.scoreProjections(accuracies, matches, misses, test_projections, self.metadata, self.cellTypeColumn, specification_value=specification_value)
            del test_data
            del test_processed
            del test_projections
        for key, value in accuracies.items():
            # print("{}: {}".format(key, value/len(test_IDs)))
            print("{}: {}".format(key, value / (10 * len(split_IDs))))

        self.testResults = (accuracies, matches, misses)
        return self.testResults

    # Get the metrics for a given projection. Optionally adjust the minimum accuracy threshold
    def scoreProjections(self, accuracies, matches, misses, projections, specification_value=0.1): # cells with maximum projection under specification_value are considered "unspecified"

        predicted_labels = []
        predicted_labels_specified = []
        true_labels = []
        matches = {}
        misses = {}
        print("Scoring projection...")
        for sample_id, sample_projections in projections.items():
            types_sorted_by_projections = sample_projections.sort_values(ascending=False).index
            true_type = self.metadata.loc[sample_id, self.cellTypeColumn]
            true_labels += [true_type]
            top_type = types_sorted_by_projections[0]
            predicted_labels += [top_type]

            if sample_projections.max() < specification_value:
                predicted_labels_specified += ['Unspecified']
                accuracies['unspecified'] += 1
            else:
                predicted_labels_specified += [top_type]

            if top_type == true_type:
                accuracies['top1'] += 1
                if true_type not in matches:
                    matches[true_type] = 1
                else:
                    matches[true_type] += 1
            else:
                if true_type not in misses:
                    misses[true_type] = 1
                else:
                    misses[true_type] += 1
            if true_type in types_sorted_by_projections[:3]:
                accuracies['top3'] += 1

        return accuracies, matches, misses

    # Create correlation matrix between cell types of basis, helpful to determine if any features are overlapping
    def getBasisCorrelations(self):
        if self.basis is None:
            print("Basis must be set first!")
            return None
        self.corr = self.basis.corr()
        return self.corr


# Get the info needed to load the specific dataset
def getDatasetSpecificInfo(dataset):
    timeColumn = None
    toExclude = []
    duplicates = False
    raw = False
    layer = None

    if dataset == "Kostas":
        filePath = "/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Kostas/Kostas.h5ad"
        cellTypeColumn = "cell_type_epithelial_mesenchymal_final"
        toKeep = ["AT1", "AT2", "AT2 activated", "Proliferating", "Transitional epithelial"]
    elif dataset == "Riemondy":
        filePath = "/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Riemondy.h5ad"
        cellTypeColumn = "labeled_clusters"
        toKeep =  ["Basal", "Injured Type II", "Naive Type I", "Naive Type II", "Transdifferentiating Type II", "Cell Cycle Arrest Type II", "Proliferating Type II"]
    elif dataset == "Strunz":
        filePath = "/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/OutsidePaperObjects/StrunzAnnData.h5ad"
        cellTypeColumn = "cell_type"
        toKeep = ["AT2", "AT2 activated", "Krt8+ ADI", "AT1", "Basal", "Mki67+ Proliferation"]
        timeColumn = "time_point"
    elif dataset == "Choi":
        filePath = "/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Choi/celltype_1_Epi_v4_merged_AT2.h5ad"
        cellTypeColumn = "celltype_4"
        toKeep =  ["AT1", "AT2", "Primed", "Intermediate", "Cycling AT2"]
        duplicates = True
    elif dataset == "Kobayashi":
        filePath = "/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/OutsidePaperObjects/KobayashiAnnData.h5ad"
        cellTypeColumn = "cell_type"
        toKeep = ["AEC1", "AEC2", "Ctgf+", "AEC2-proliferating", "Lgals3+"]
    elif dataset == "Kathiriya":
        filePath = "/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Kathiriya/Kathiriya.h5ad"
        cellTypeColumn = "celltypes"
        toKeep = [0, 1, 2, 3, 4, 5, 6]
    elif dataset == "Habermann":
        filePath = "/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Habermann/Habermann.h5ad"
        cellTypeColumn = "celltype"
        toKeep = ["AT1", "AT2", "KRT5-/KRT17+", "Proliferating Epithelial Cells", "Transitional AT2", "SCGB3A2+", "SCGB3A2+ SCGB1A1+"]
    elif dataset == "Bibek":
        filePath = "/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/BibekPneumonectomy/objects/Bibek.h5ad"
        cellTypeColumn = "annotation_update"
        toKeep = ["AT1", "AT2", "Krt8 high AT2", "Activated AT2", "Ciliated", "Proliferating AT2", "Secretory"]
        timeColumn = "days"
    elif dataset == "Rawlins":
        filePath = '/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Andrea/EmmaRawlinsBasis.h5ad'
        cellTypeColumn = 'new_celltype'
        toKeep = ["AT1", "AT2", "Krt8 high AT2", "Activated AT2", "Ciliated", "Proliferating AT2", "Secretory"]
    elif dataset == "Tsukui":
        filePath = '/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/BibekPneumonectomy/objects/Tsukui.h5ad'
        cellTypeColumn = "celltype"
        toKeep = []
    elif dataset == "Burgess":
        filePath = '/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Burgess/Burgess.h5ad'
        cellTypeColumn = "new_cluster"
        # cell_type_column = "seurat_clusters"
        toKeep = ["CASP4+ cells", "Differentiating iAT2/iAT1", "Early iAT1s", "FGL1 high iAT2s", "iAT2s", "Late iAT1s"]

    return cellTypeColumn, toKeep, toExclude, filePath, timeColumn, duplicates, raw, layer