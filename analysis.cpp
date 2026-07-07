#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <climits>
#include <string>
#include <chrono>
#include <iomanip> // For formatting output

using namespace std;

// Global Constants 
const int N_POINTS = 5000;      // Number of points in the dataset
const int D_DIMENSIONS = 128;   // Dimensions (d)
const int K_BITS = 10;          
const int L_TABLES = 5;         
const int NUM_TEST_QUERIES = 100; // Number of queries to run for evaluation

// Data Structures

// Represents a d-dimensional binary vector
struct Point {
    int id;
    vector<int> data; // Stores 0s and 1s

    // Calculate Hamming Distance between two points
    static int hammingDistance(const Point& a, const Point& b) {
        int dist = 0;
        for (size_t i = 0; i < a.data.size(); ++i) {
            if (a.data[i] != b.data[i]) {
                dist++;
            }
        }
        return dist;
    }
};

// LSH Class Definition
class LSH {
private:
    int d; // Dimensions
    int k; // Number of bits to sample per hash function
    int L; // Number of hash tables (families)

    // Each table has a unique "Hash Function" defined by 'k' random indices.
    // randomIndices[table_index][bit_index]
    vector<vector<int>> randomIndices;

    // The Hash Tables. 
    // Key: The hash signature (string). Value: List of Point IDs.
    vector<unordered_map<string, vector<int>>> tables;

    // Pointer to the actual data (to save memory)
    const vector<Point>* dataset;

public:
    LSH(int dimensions, int k_complexity, int l_tables) 
        : d(dimensions), k(k_complexity), L(l_tables) {
        
        tables.resize(L);
        
        // Initialize Random Number Generator
        mt19937 rng(42); // Fixed seed for reproducibility
        uniform_int_distribution<int> dist(0, d - 1);

        // Generate Family H (L different sets of K indices)
        for (int i = 0; i < L; ++i) {
            vector<int> indices;
            for (int j = 0; j < k; ++j) {
                indices.push_back(dist(rng));
            }
            randomIndices.push_back(indices);
        }
    }

    // Compute hash signature for a specific table
    string computeHash(const Point& p, int tableIdx) {
        string signature = "";
        const vector<int>& indices = randomIndices[tableIdx];
        
        for (int idx : indices) {
            signature += to_string(p.data[idx]);
        }
        return signature;
    }

    // Index the dataset
    void train(const vector<Point>& points) {
        dataset = &points;
        cout << "[LSH] Indexing " << points.size() << " points across " << L << " tables..." << endl;

        for (const Point& p : points) {
            for (int i = 0; i < L; ++i) {
                string hashVal = computeHash(p, i);
                tables[i][hashVal].push_back(p.id);
            }
        }
    }

    // Method B: Standard LSH (Union of all L tables)
    int findNearestNeighbor_Union(const Point& query, int& outDistance, int& candidatesChecked) {
        int bestPointId = -1;
        int minDist = INT_MAX;
        candidatesChecked = 0;

        // Visited array to prevent checking the same candidate multiple times
        vector<bool> visited(dataset->size(), false);

        for (int i = 0; i < L; ++i) {
            string queryHash = computeHash(query, i);

            if (tables[i].count(queryHash)) {
                const vector<int>& bin = tables[i].at(queryHash);
                
                for (int candidateId : bin) {
                    if (visited[candidateId]) continue;
                    visited[candidateId] = true;
                    candidatesChecked++;

                    int dist = Point::hammingDistance(query, (*dataset)[candidateId]);
                    if (dist < minDist) {
                        minDist = dist;
                        bestPointId = candidateId;
                    }
                }
            }
        }
        outDistance = minDist;
        return bestPointId;
    }

    // Method C: Stochastic LSH (Single Random Hash Function from H)
    int findNearestNeighbor_SingleRandom(const Point& query, int& outDistance, int& candidatesChecked) {
        int bestPointId = -1;
        int minDist = INT_MAX;
        candidatesChecked = 0;

        // Pick one hash function (table) uniformly at random
        random_device rd;
        mt19937 rng(rd());
        uniform_int_distribution<int> dist(0, L - 1);
        int randomTableIndex = dist(rng);

        string queryHash = computeHash(query, randomTableIndex);

        if (tables[randomTableIndex].count(queryHash)) {
            const vector<int>& bin = tables[randomTableIndex].at(queryHash);
            
            for (int candidateId : bin) {
                // No visited check needed for single bin
                candidatesChecked++;

                int dist = Point::hammingDistance(query, (*dataset)[candidateId]);
                if (dist < minDist) {
                    minDist = dist;
                    bestPointId = candidateId;
                }
            }
        }
        outDistance = minDist;
        return bestPointId;
    }
};

// Helper Functions

// Method A: Brute Force Search (Ground Truth)
int bruteForceNN(const vector<Point>& data, const Point& query, int& outDist) {
    int bestId = -1;
    int minDist = INT_MAX;

    for (const Point& p : data) {
        int dist = Point::hammingDistance(query, p);
        if (dist < minDist) {
            minDist = dist;
            bestId = p.id;
        }
    }
    outDist = minDist;
    return bestId;
}

// Generate random dataset
vector<Point> generateDataset(int n, int d) {
    vector<Point> data(n);
    mt19937 rng(random_device{}());
    uniform_int_distribution<int> dist(0, 1);

    for (int i = 0; i < n; ++i) {
        data[i].id = i;
        data[i].data.resize(d);
        for (int j = 0; j < d; ++j) {
            data[i].data[j] = dist(rng);
        }
    }
    return data;
}

// Generate a single random point
Point generateRandomPoint(int d) {
    Point p;
    p.id = -1;
    p.data.resize(d);
    mt19937 rng(random_device{}());
    uniform_int_distribution<int> dist(0, 1);
    for(int j=0; j<d; ++j) p.data[j] = dist(rng);
    return p;
}

// Evaluation Logic
void evaluatePerformance(LSH& lsh, const vector<Point>& dataset, int numTestQueries) {
    cout << "\n================================================================================" << endl;
    cout << "  PERFORMANCE EVALUATION: Comparing Methods A, B, and C over " << numTestQueries << " Random Queries" << endl;
    cout << "================================================================================" << endl;

    // Accumulators for statistics
    double totalTimeBF = 0;
    
    // Method B (Union) Stats
    double totalTimeUnion = 0;
    long totalCandidatesUnion = 0;
    int correctMatchesUnion = 0; // Exact matches
    double sumRatioUnion = 0.0;  // For Approx Ratio

    // Method C (Single) Stats
    double totalTimeSingle = 0;
    long totalCandidatesSingle = 0;
    int correctMatchesSingle = 0;
    double sumRatioSingle = 0.0;

    for (int i = 0; i < numTestQueries; ++i) {
        Point Q = generateRandomPoint(D_DIMENSIONS);

        // Run Brute Force (Method A) ---
        int bfDist;
        chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
        bruteForceNN(dataset, Q, bfDist);
        chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
        totalTimeBF += chrono::duration_cast<chrono::microseconds>(t2 - t1).count();

        // Avoid division by zero in ratios (though unlikely in high dim)
        double safeBFDist = (bfDist == 0) ? 0.1 : (double)bfDist;

        // Run LSH Union (Method B) ---
        int unionDist, unionCand;
        t1 = chrono::high_resolution_clock::now();
        int unionId = lsh.findNearestNeighbor_Union(Q, unionDist, unionCand);
        t2 = chrono::high_resolution_clock::now();
        
        totalTimeUnion += chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        totalCandidatesUnion += unionCand;
        
        if (unionId != -1) {
             if (unionDist == bfDist) correctMatchesUnion++;
             sumRatioUnion += ((double)unionDist / safeBFDist);
        } else {
            // Penalize missed search heavily in ratio (e.g., consider it infinite or just skip)
            // For simple average, we'll assume a penalty factor or just add 0 and track misses separately.
            // Here we add a penalty factor of 2.0 for simplicity in the average.
            sumRatioUnion += 2.0; 
        }

        // Run LSH Single Random (Method C) ---
        int singleDist, singleCand;
        t1 = chrono::high_resolution_clock::now();
        int singleId = lsh.findNearestNeighbor_SingleRandom(Q, singleDist, singleCand);
        t2 = chrono::high_resolution_clock::now();

        totalTimeSingle += chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        totalCandidatesSingle += singleCand;
        
        if (singleId != -1) {
            if (singleDist == bfDist) correctMatchesSingle++;
            sumRatioSingle += ((double)singleDist / safeBFDist);
        } else {
            sumRatioSingle += 2.0; // Penalty for miss
        }
    }

    // Calculate Averages
    double avgTimeBF = totalTimeBF / numTestQueries;
    
    double avgTimeUnion = totalTimeUnion / numTestQueries;
    double avgCandUnion = (double)totalCandidatesUnion / numTestQueries;
    double accuracyUnion = (double)correctMatchesUnion / numTestQueries * 100.0;
    double avgRatioUnion = sumRatioUnion / numTestQueries;

    double avgTimeSingle = totalTimeSingle / numTestQueries;
    double avgCandSingle = (double)totalCandidatesSingle / numTestQueries;
    double accuracySingle = (double)correctMatchesSingle / numTestQueries * 100.0;
    double avgRatioSingle = sumRatioSingle / numTestQueries;

    // Print Results Table
    cout << left << setw(25) << "Metric" 
         << setw(20) << "[A] Brute Force" 
         << setw(20) << "[B] LSH Union" 
         << setw(20) << "[C] LSH Single" << endl;
    cout << string(85, '-') << endl;

    cout << left << setw(25) << "Avg Time (microsec)" 
         << setw(20) << avgTimeBF
         << setw(20) << avgTimeUnion
         << setw(20) << avgTimeSingle << endl;

    cout << left << setw(25) << "Avg Candidates Checked" 
         << setw(20) << N_POINTS
         << setw(20) << avgCandUnion
         << setw(20) << avgCandSingle << endl;

    cout << left << setw(25) << "Exact Recall (%)" 
         << setw(20) << "100%"
         << setw(20) << accuracyUnion 
         << setw(20) << accuracySingle << endl;
         
    cout << left << setw(25) << "Approx Ratio (1.0=Best)" 
         << setw(20) << "1.00"
         << setw(20) << avgRatioUnion 
         << setw(20) << avgRatioSingle << endl;
    
    cout << string(85, '-') << endl;

    // Interpretation
    cout << "\nInterpretation:" << endl;
    cout << "1. Exact Recall: % of times we found the EXACT nearest neighbor." << endl;
    cout << "2. Approx Ratio: Average of (Found Dist / True Dist). 1.05 means the found" << endl;
    cout << "   neighbor is on average 5% further away than the true neighbor." << endl;
}




int main() {
	cout<<"\n";
    cout << "Configuration: N=" << N_POINTS << ", D=" << D_DIMENSIONS 
         << ", K=" << K_BITS << ", L=" << L_TABLES << endl;
	cout<<"\n";




	/*

	// Individual Output:
    // Generate Data & Train
    cout << "Generating Dataset X" << endl;
    vector<Point> X = generateDataset(N_POINTS, D_DIMENSIONS);

    Point queryQ; // Generate random query
    queryQ.id = -1;
    queryQ.data.resize(D_DIMENSIONS);
    mt19937 rng(random_device{}());
    uniform_int_distribution<int> dist(0, 1);
    for(int j=0; j<D_DIMENSIONS; ++j) queryQ.data[j] = dist(rng);

    cout << "Building LSH Index" << endl;
    LSH lshIndex(D_DIMENSIONS, K_BITS, L_TABLES);
    
    chrono::high_resolution_clock::time_point startBuild = chrono::high_resolution_clock::now();
    lshIndex.train(X);
    chrono::high_resolution_clock::time_point endBuild = chrono::high_resolution_clock::now();
    
    cout << "   Build Time: " 
         << chrono::duration_cast<chrono::milliseconds>(endBuild - startBuild).count() 
         << " ms" << endl;

    cout << "\nRunning Search Algorithms" << endl;

    //Brute Force (Ground Truth)
    int exactDist;
    chrono::high_resolution_clock::time_point startBF = chrono::high_resolution_clock::now();
    int trueNN = bruteForceNN(X, queryQ, exactDist);
    chrono::high_resolution_clock::time_point endBF = chrono::high_resolution_clock::now();

    // LSH (Standard - Union of L Tables)
    int lshDist;
    int lshCandidates;
    chrono::high_resolution_clock::time_point startLSH = chrono::high_resolution_clock::now();
    int lshNN = lshIndex.findNearestNeighbor_Union(queryQ, lshDist, lshCandidates);
    chrono::high_resolution_clock::time_point endLSH = chrono::high_resolution_clock::now();

    //LSH (Single Random Hash Function)
    int singleDist;
    int singleCandidates;
    chrono::high_resolution_clock::time_point startSingle = chrono::high_resolution_clock::now();
    int singleNN = lshIndex.findNearestNeighbor_SingleRandom(queryQ, singleDist, singleCandidates);
    chrono::high_resolution_clock::time_point endSingle = chrono::high_resolution_clock::now();


    //Output & Comparison
    cout << "\nResults" << endl;
    
    cout << "[A] Brute Force (Ground Truth):" << endl;
    cout << "    Nearest ID: " << trueNN << endl;
    cout << "    Distance:   " << exactDist << endl;
    cout << "    Time:       " << chrono::duration_cast<chrono::microseconds>(endBF - startBF).count() << " us" << endl;

    cout << "\n[B] LSH (Union of " << L_TABLES << " Tables):" << endl;
    if (lshNN != -1) {
        cout << "    Nearest ID: " << lshNN << endl;
        cout << "    Distance:   " << lshDist << " (Error: " << (lshDist - exactDist) << ")" << endl;
        cout << "    Time:       " << chrono::duration_cast<chrono::microseconds>(endLSH - startLSH).count() << " us" << endl;
        cout << "    Candidates: " << lshCandidates << endl;
    } else {
        cout << "    No neighbor found in any bin." << endl;
    }

    cout << "\n[C] LSH (Single Random Hash Function):" << endl;
    if (singleNN != -1) {
        cout << "    Nearest ID: " << singleNN << endl;
        cout << "    Distance:   " << singleDist << " (Error: " << (singleDist - exactDist) << ")" << endl;
        cout << "    Time:       " << chrono::duration_cast<chrono::microseconds>(endSingle - startSingle).count() << " us" << endl;
        cout << "    Candidates: " << singleCandidates << endl;
    }
	else {
        cout << "    No neighbor found (Query hashed to an empty bin)." << endl;
    }

	*/



    
	///*


	// Evaluation Output:
    // 1. Generate Data
    cout << "\nGenerating Dataset X (" << N_POINTS << " binary vectors)" << endl;
    vector<Point> X = generateDataset(N_POINTS, D_DIMENSIONS);

    // 2. Train LSH
    cout << "Building LSH Index" << endl;
    LSH lshIndex(D_DIMENSIONS, K_BITS, L_TABLES);
    
    chrono::high_resolution_clock::time_point startBuild = chrono::high_resolution_clock::now();
    lshIndex.train(X);
    chrono::high_resolution_clock::time_point endBuild = chrono::high_resolution_clock::now();
    
    cout << "   Index Build Time: " 
         << chrono::duration_cast<chrono::milliseconds>(endBuild - startBuild).count() 
         << " ms" << endl;

    // 3. Evaluate
    evaluatePerformance(lshIndex, X, NUM_TEST_QUERIES);

	//*/
}
