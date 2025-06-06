import statistics
import numpy as np
import pandas as pd
import math
from scipy import signal


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

    for i in range(geneCount - 1):
        gene1Series = filterDFByGenes(cluster, cluster.index[i])
        for j in range(i + 1, geneCount):
            gene2Series = filterDFByGenes(cluster, cluster.index[j])

            corr = gene1Series.corr(gene2Series)
            if math.isnan(corr):
                continue
            else:
                correlationList.append(abs(corr))

    return sum(correlationList) / geneCount**2 #, statistics.mean(correlationList)


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
        if i % 500 == 0:
            print(i)

        geneOutSeries = filterDFByGenes(outDF, geneOut)

        for geneIn in inDF.index:
            geneInSeries = filterDFByGenes(inDF, geneIn)
            corr = geneOutSeries.corr(geneInSeries)
            if not math.isnan(corr):
                correlationList.append(abs(corr))

    return sum(correlationList) / (len(inDF.index) * (len(df.index) - len(inDF.index))) #correlationList, len(df.index)


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


def getGeneEntropy(df, gene)