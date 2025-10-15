import statistics
import numpy as np
import pandas as pd
import math
from scipy import signal
import scipy.cluster.hierarchy as spc
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

def filterDFByTime(df, timeName, timeValues):
    return df.loc[:, timeValues == timeName]


def filterDFByGenes(df, genes, negative=False):
    if type(genes) is list:
        if negative:
            return df.loc[~ df.index.isin(genes), :]
        return df.loc[df.index.isin(genes), :]
    return df.loc[genes, :]


def filterDFByGeneExpression(df, gene, threshold=1):
    return df.loc[:, df.loc[gene, :] > threshold]


def filterDFByMinimumGeneExpression(df, includeCriteria=None, threshold=0):
    if includeCriteria is not None:
        df = df.loc[:, includeCriteria]
    return df.loc[[val > threshold for val in df.mean(axis=1)], :]


def filterDFByGeneVariance(df, threshold=0.01):
    return df.loc[df.var(axis=1) > threshold, :]


def filterDFByTimeAndCluster(df, timeName, timeValues, clusterName, clusterValues, negative=False):
    if negative:
        return df.loc[:, np.logical_and(clusterValues != clusterName, timeValues == timeName)]
    return df.loc[:, np.logical_and(clusterValues == clusterName, timeValues == timeName)]


def filterDFByTimeAndGeneExpression(df, gene, timeName, timeValues, threshold=1, negative=False):
    if negative:
        return df.loc[:, np.logical_and(df.loc[gene, :] <= threshold, timeValues == timeName)]
    return df.loc[:, np.logical_and(df.loc[gene, :] > threshold, timeValues == timeName)]


def getClusterCoefficientOfVariation(cluster):
    CVList = []
    for gene in cluster.index:
        geneArray = cluster.loc[gene, :]
        mean = statistics.mean(geneArray)
        sd = np.std(geneArray)
        CVList.append(sd / mean)
    return statistics.mean(CVList)


def getClusterStandardDeviation(cluster, timePoint=None):
    return statistics.mean(cluster.std(axis=1)) # Standard deviation of each row in df, i.e. of each gene


def getInternalCorrelationOneToMany(cluster, gene, timePoint=None):
    correlationList = []
    mainGeneSeries = filterDFByGene(cluster, gene)
    for otherGene in cluster.index:
        if otherGene != gene:
            corr = mainGeneSeries.corr(filterDFByGene(cluster, otherGene))
            if math.isnan(corr):
                continue
            else:
                correlationList.append(abs(corr))

    return correlationList, statistics.mean(correlationList)


def getInternalCorrelationManyToMany(cluster):
    correlationList = []
    geneCount = len(cluster.index)

    for i in range(geneCount):
        gene1Series = filterDFByGenes(cluster, cluster.index[i])
        for j in range(geneCount):
            if i == j:
                # continue
                pass
            gene2Series = filterDFByGenes(cluster, cluster.index[j])

            corr = gene1Series.corr(gene2Series)
            if math.isnan(corr):
                continue
            else:
                correlationList.append(abs(corr))

    # return sum(correlationList) / geneCount
    return sum(correlationList) / geneCount**2


def getExternalCorrelationOneToMany(df, timeName, timeValues, geneOfInterest,
                                     clusterName=None, clusterValues=None,
                                     clusterGene=None, expressionThreshold=1,
                                     varianceThreshold=None):

    if varianceThreshold is not None:
        print("Filtering by gene variance...")
        df = filterDFByGeneVariance(df, varianceThreshold)

    if clusterGene is not None:
        print("Filtering by time and gene expression...")
        inDF = filterDFByTimeAndGeneExpression(df, clusterGene, timeName, timeValues, threshold=expressionThreshold, negative=False)
        outDF = filterDFByTimeAndGeneExpression(df, clusterGene, timeName, timeValues, threshold=expressionThreshold, negative=True)
    elif clusterName is not None and clusterValues is not None:
        print("Filtering by time and cluster...")
        inDF = filterDFByTimeAndCluster(df, timeName, timeValues, clusterName, clusterValues, negative=False)
        outDF = filterDFByTimeAndCluster(df, timeName, timeValues, clusterName, clusterValues, negative=True)
    else:
        print("No cluster selected as input!")
        return None, None

    print("Filtering by gene of interest...")
    mainGeneSeries = filterDFByGene(inDF, geneOfInterest)
    print(mainGeneSeries)
    
    correlationList = []
    i = 0
    print("Calculating correlations...")
    for geneOut in outDF.index:
        # print(geneOut)
        i += 1
        if i % 100 == 0:
            print(i)
        if i > 3:
            break
        geneOutSeries = filterDFByGenes(outDF, geneOut)
        # print(geneOutSeries)
        corr = signal.correlate(mainGeneSeries, geneOutSeries)

        # corr = mainGeneSeries.corr(geneOutSeries)
        print(corr)
        # return [mainGeneSeries, geneOutSeries]
        if not math.isnan(corr):
            correlationList.append(abs(corr))

    return correlationList


def getExternalCorrelationManyToMany(df, timeName, timeValues,
                                     DNB=None,
                                     clusterName=None, clusterValues=None,
                                     gene=None, expressionThreshold=1,
                                     varianceThreshold=None):

    if varianceThreshold is not None:
        print("Filtering by gene variance...")
        df = filterDFByGeneVariance(df, varianceThreshold)

    if DNB is not None:
        print("Filtering by DNB...")
        inDF = filterDFByGenes(df, DNB, negative=False)
        outDF = filterDFByGenes(df, DNB, negative=True)
    elif gene is not None:
        print("Filtering by time and gene expression...")
        inDF = filterDFByTimeAndGeneExpression(df, gene, timeName, timeValues, threshold=expressionThreshold, negative=False)
        outDF = filterDFByTimeAndGeneExpression(df, gene, timeName, timeValues, threshold=expressionThreshold, negative=True)
    elif clusterName is not None and clusterValues is not None:
        print("Filtering by time and cluster...")
        inDF = filterDFByTimeAndCluster(df, timeName, timeValues, clusterName, clusterValues, negative=False)
        outDF = filterDFByTimeAndCluster(df, timeName, timeValues, clusterName, clusterValues, negative=True)
    else:
        print("No cluster selected as input!")
        return None, None

    correlationList = []
    i = 0
    print("Calculating correlations...")
    for geneOut in outDF.index:
        i += 1
        if i % 1000 == 0:
            print(i)

        geneOutSeries = filterDFByGenes(outDF, geneOut)

        for geneIn in inDF.index:
            geneInSeries = filterDFByGenes(inDF, geneIn)
            corr = geneOutSeries.corr(geneInSeries)
            if not math.isnan(corr):
                correlationList.append(abs(corr))

    # return sum(correlationList) / len(df.index)
    return sum(correlationList) / (len(inDF.index) * len(outDF.index))


def getIScore(cluster, clusterList, gene, timePoint=None):
    cov = getClusterCovariance(cluster, timePoint=timePoint)
    internalCorr = getInternalCorrelation(cluster, gene, timePoint=timePoint)
    externalCorr = getExternalCorrelation(clusterList, gene, timePoint=timePoint)
    return cov * internalCorr / externalCorr


def getSummaryValue(df, timeName, timeValues, DNB, varianceThreshold=0.1, summaryType="CI"):
    df = filterDFByTime(df, timeName, timeValues)
    df = filterDFByGeneVariance(df, varianceThreshold)
    cluster = filterDFByGenes(df, DNB)
    if summaryType == "CI":
        toReturn = getClusterStandardDeviation(cluster)
    elif summaryType == "CV":
        toReturn = getClusterCoefficientOfVariation(cluster)
    internalCorr = getInternalCorrelationManyToMany(cluster)
    externalCorr = getExternalCorrelationManyToMany(df, timeName, timeValues, DNB=DNB)
    return [toReturn * internalCorr / externalCorr, toReturn, internalCorr, externalCorr]


def getCellEntropy():
    pass


def getGeneEntropy(df, gene, timeName=None, timeValues=None):
    if timeName is not None and timeValues is not None:
        geneSeries = df.loc[gene, timeValues==timeName]
    else:
        geneSeries = filterDFByGenes(df, gene)

    # totalGeneCount = np.sum(geneSeries)
    # print(totalGeneCount)
    # if totalGeneCount == 0:
    #     return 0
    geneSeriesLength = len(geneSeries)

    hist, bins = np.histogram(geneSeries, bins=geneSeriesLength // 2)
    totalEntropy = 0
    for sample in hist:
        if sample != 0:
            # p = sample / totalGeneCount
            # p = sample / (binCount)
            p = sample / geneSeriesLength
            totalEntropy += math.log2(p) * p
    return -1 * totalEntropy


def getDifferentiallyExpressedGenes(annObject, differentialColumn, case, individualCompare=False, useRaw=False, includeCriteria=None):
    annObjectCopy = annObject.copy()
    annObjectCopy = annObjectCopy if includeCriteria is None else annObjectCopy[includeCriteria]
    annotations = annObjectCopy.obs[differentialColumn]
    annotations = annotations.cat.remove_unused_categories()
    print("Normalizing...")
    sc.pp.normalize_total(annObjectCopy, inplace=True)
    # annObjectCopy = sc.pp.normalize_total(annObjectCopy, inplace=False)
    sc.pp.log1p(annObjectCopy, copy=False)
    # annObjectCopy = sc.pp.log1p(annObjectCopy, copy=True)
    
    if individualCompare:
        diffTableMap = {}
        for comparisonGroup in annotations.cat.categories:
            print("Currently comparing against " + comparisonGroup)
            if comparisonGroup != case:
                filteredAnnObject = annObjectCopy[annotations.isin([case, comparisonGroup])].copy()
                sc.tl.rank_genes_groups(filteredAnnObject, differentialColumn, method='wilcoxon', use_raw=useRaw, copy=False)
                diffTableMap[comparisonGroup] = sc.get.rank_genes_groups_df(filteredAnnObject, group=case).set_index("names")
        print("Done!")
        return diffTableMap

    sc.tl.rank_genes_groups(annObjectCopy, differentialColumn, method='wilcoxon', use_raw=useRaw, copy=False)
    diffTable = sc.get.rank_genes_groups_df(annObjectCopy, group=case).set_index("names")
    # reducedDiff = diffTable.loc[np.logical_and(diffTable['pvals_adj'] < 0.05, abs(diffTable['logfoldchanges']) > 2), :]
    print("Complete")
    return diffTable, annObjectCopy


# Given a DEG table, filter table to genes meeting certain conditions
def getTopGenes(diffTable, minimumFoldChange=2, outFile=None, surfaceGeneFile="/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Transdifferentiation/CellSurfaceGenes.txt", checkSurface=False, requireOverexpression=False):
    
    # Set conditions for differential expression
    logFoldChanges = diffTable['logfoldchanges'] if requireOverexpression else diffTable['logfoldchanges'].abs()
    conditions = [diffTable['pvals_adj'] < 0.05, logFoldChanges > minimumFoldChange]

    # Set condition for surface genes
    if checkSurface:
        cellSurfaceGenes = pd.read_csv(surfaceGeneFile)
        conditions.append(diffTable.index.isin(cellSurfaceGenes.columns))

    # Apply all conditions
    combinedCondition = conditions[0]
    for i in range(1, len(conditions)):
        combinedCondition = np.logical_and(combinedCondition, conditions[i])
    filteredGenes = diffTable.loc[combinedCondition, :]

    if outFile is not None:
        filteredGenes.to_csv(outFile)
    return filteredGenes


# Given DEG between a group and each other, get table of genes with high expression and universal high differential expression
def getCombinedTopGenes(diffTableMap, df, includeCriteria=None, expressionThreshold=1, missesAllowed=0, outFile=None, checkSurface=False, requireOverexpression=False):
    if includeCriteria is not None: # Usually filter to the case
        df = df.loc[:, includeCriteria]

    allGenes = pd.DataFrame(index=df.index)
    comparisonGroups = list(diffTableMap.keys())
    for group in comparisonGroups:
        validGenes = getTopGenes(diffTableMap[group], checkSurface=checkSurface, requireOverexpression=requireOverexpression).index
        valid = [int(val in validGenes) for val in df.index]
        allGenes[group] = valid

    # allGenes.set_index("Gene", inplace=True)
    totalValid = allGenes.sum(axis=1)
    allGenes["Successes"] = totalValid

    filteredGenes = df.loc[allGenes["Successes"] > len(comparisonGroups) - missesAllowed - 1, :]
    filteredGenes = filterDFByMinimumGeneExpression(filteredGenes, threshold=expressionThreshold)
    targetDF = pd.DataFrame({"Average Normalized Expression": filteredGenes.mean(axis=1)}, index=filteredGenes.index)
    for group in comparisonGroups:
        DEG = diffTableMap[group].loc[:, "logfoldchanges"].copy().rename("logfoldchanges " + group)
        targetDF = targetDF.join(DEG, how="left", rsuffix=group)

    if outFile is not None:
        targetDF.to_csv(outFile)
    if missesAllowed > 0:
        targetDF = targetDF.join(allGenes["Successes"], how="left")
    return targetDF


def clusterGenesByCorrelation(df):
    # Get correlation "distances"
    corr = df.T.corr().values
    pdist_uncondensed = 1.0 - abs(corr)
    pdist_condensed = np.concatenate([row[i+1:] for i, row in enumerate(pdist_uncondensed)])

    # Cluster based on these distances
    linkage = spc.linkage(pdist_condensed, method='complete')
    idx = spc.fcluster(linkage, 0.5 * pdist_condensed.max(), 'distance')

    # Create map of clusters to their genes
    clusterDict = {}
    for i in range(len(idx)):
        cluster = int(idx[i])
        gene = df.index.iloc[i]
        if cluster not in clusterDict.keys():
            clusterDict[cluster] = [gene]
        else:
            clusterDict[cluster].append(gene)
    return clusterDict


def getDominantGroups(df, clustersList, timeValues, timesSorted, differentialColumn, case, control):

    print("Normalizing...")
    normalizedDF = df.copy()
    for gene in df.index:
        geneControl = df.loc[df.index == gene, differentialColumn == control]
        geneCase = df.loc[df.index == gene, differentialColumn == case]
        meanControl = statistics.mean(geneControl)
        sdControl = np.std(geneControl)
        normalizedDF.loc[normalizedDF.index == gene, :] = (geneCase - meanControl) / sdControl

    print("Finding DNBs...")
    clusterValues = {}
    for time in timesSorted:
        timeDF = filterDFByTime(normalizedDF, time, timeValues)
        clusterValues[time] = {}
        for clusterList in clustersList:
            # Ideally screen for requirements first or also
            if len(clusterList) > 2:
                for cluster in clusterList:
                    genes = clusterList[cluster]
                    clusterValues[time][genes] = getSummaryValue(timeDF, time, timeValues, genes, summaryType="CI")

    return clusterValues


def findDNB(annObject, timeColumnName, timesSorted, differentialColumn, case, control):
    annObjectCopy = annObject.copy()
    metadata = annObjectCopy.obs
    annObjectCopy = annObjectCopy[np.logical_or(metadata[differentialColumn] == case, metadata[differentialColumn] == control)]
    metadata = annObjectCopy.obs
    df = annObjectCopy.to_df().T

    print("Clustering...")
    clustersList = []
    for time in timesSorted:
        print(time)
        timeDF = df.loc[:, metadata[timeColumnName] == time]
        timeAnnObject = annObjectCopy.copy()
        timeAnnObject = timeAnnObject[timeDF.columns, :]
        timeDF = timeAnnObject.to_df().T
        highVarianceGenes = timeDF.loc[timeDF.var(axis=1) > 0.01, :].index
        timeAnnObject = timeAnnObject[:, list(highVarianceGenes)] # Filter by variance
        timeDF = timeAnnObject.to_df().T
        genesOfInterest = getDifferentiallyExpressedGenes(timeAnnObject, differentialColumn, case)
        clustersList.append(clusterGenesByCorrelation(timeDF.loc[timeDF.index.isin(genesOfInterest), :]))

    print("Finding DNB...")
    rankedGroups = getDominantGroups(df, clustersList, metadata[timeColumnName], timesSorted, differentialColumn, case, control)
    print("Done!")
    return rankedGroups


# Get distance of each point to mean of cluster 
def getClusterDistances(df):
    mean = df.mean(axis=1)
    distances = np.linalg.norm(df.values - np.tile(mean, (len(df.columns), 1)).T, axis=0)
    return distances, mean


# Filter dataframe of n-dimensional points down to x % of data closest to the mean 
def getQuantileData(df, includeCriteria=None, quantile=0.8, quantileMin=None, minSamples=50):
    if includeCriteria is not None:
        df = df.loc[:, includeCriteria]
    if len(df.columns) < minSamples:
        return None
    distances, mean = getClusterDistances(df)
    upperQuantile = np.quantile(distances, quantile)
    lowerQuantile = 0 if quantileMin is None else np.quantile(distances, quantileMin)
    return df.loc[:, np.logical_and(distances < upperQuantile, distances > lowerQuantile)]


# Find shortest distance between n-dimensional points
def getMinimumDistance(df1, df2, metric='euclidean'):
    distances = cdist(df1.values.T, df2.values.T, metric=metric)
    return np.min(distances), np.unravel_index(np.argmin(distances), distances.shape)


# Get mean distance between two clusters
def getMeanDistance(df1, df2):
    mean1 = df1.mean(axis=1)
    mean2 = df2.mean(axis=1)
    return (np.linalg.norm(mean1 - mean2), (0, 0))


# Get DataFrame of minimal distances between clusters given dict of cluster names to points
def getDistanceMap(clusterMap, method="min"):
    # Select distance metric
    if method == "min":
        distanceFunc = getMinimumDistance
    elif method == "mean":
        distanceFunc = getMeanDistance
    else:
        print("Invalid distance method selected!")
        return None

    # Initialize dict of distances between states
    stateDistancesMap = {}
    stateList = sorted(clusterMap.keys())
    stateCount = len(stateList)
    for state in stateList:
        stateDistancesMap[state] = {}
        stateDistancesMap[state][state] = 0

    # Set distance between each pair of clusters
    for i in range(0, stateCount - 1):
        state1 = stateList[i]
        for j in range(i + 1, stateCount):
            state2 = stateList[j]
            stateDistancesMap[state1][state2] = stateDistancesMap[state2][state1] = distanceFunc(clusterMap[state1], clusterMap[state2])[0]
    distances = pd.DataFrame.from_dict(stateDistancesMap, orient="index")
    return distances


# Using state clusters filtered by quantiles, plot matrix of distances between clusters
def stateDistancePlot(topObject, projectionName, includeCriteria=None, quantile=0.8, quantileMin=None, minSamples=50, 
                      figX=8, figY=8, title="", outFile=None, method="min"):

    # Get quantiles
    stateCloseValuesMap = {}
    projection = topObject.projections[projectionName] if includeCriteria is None else topObject.projections[projectionName].loc[:, includeCriteria]
    for state in topObject.sortedCellTypes:
        reducedState = getQuantileData(projection, quantile=quantile, quantileMin=quantileMin, minSamples=minSamples, includeCriteria=topObject.annotations == state)
        if reducedState is not None:
            stateCloseValuesMap[state] = reducedState

    # Get distances
    distances = getDistanceMap(stateCloseValuesMap, method=method)
    
    # Plot results
    plt.subplots(1, 1, figsize=(figX, figY))
    labels = sorted(distances.keys())
    sns.heatmap(distances, annot=True, fmt=".2f", cmap='plasma', xticklabels=labels, yticklabels=labels,
        annot_kws={"size": 10}, cbar=True)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    if outFile is not None:
        plt.savefig(outFile)
    plt.show()


# Plot average distance to centroid of each cluster
def selfMeanDistancePlot(clusterMap):
    stateList = sorted(clusterMap.keys())
    meanDistances = []
    for state in stateList:
        distances = getClusterDistances(clusterMap[state])[0]
        meanDistances.append(sum(distances) / len(distances))
    plt.bar(stateList, meanDistances)
    plt.xticks(rotation=90)
    plt.show()


def getPCA(df):
    pca = PCA(100)
    return pca.fit_transform(df)


