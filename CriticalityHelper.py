import statistics
import numpy as np
import pandas as pd
import math
from scipy import signal
import scipy.cluster.hierarchy as spc
import scanpy as sc


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


def getDifferentiallyExpressedGenes(annObject, differentialColumn, case):
    annObjectCopy = annObject
    sc.pp.normalize_total(annObjectCopy, inplace=True)
    sc.pp.log1p(annObjectCopy, copy=False)
    sc.tl.rank_genes_groups(annObjectCopy, differentialColumn, method='t-test', use_raw=False, copy=False)
    diffTable = sc.get.rank_genes_groups_df(annObjectCopy, group=case)
    reducedDiff = diffTable.loc[np.logical_and(diffTable['pvals_adj'] < 0.05, abs(diffTable['logfoldchanges']) > 2), :]
    print("Complete")
    return reducedDiff['names']


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

