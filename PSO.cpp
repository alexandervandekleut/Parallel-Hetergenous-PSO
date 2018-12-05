#include <float.h> // for DBL_MAX
#include <iostream> // printing/debugging
#include <random> // random number generator
#include <vector> // dynamic object array with handled memory
#include <math.h> // cos
#include <mpi.h> // openMPI
#include <stdio.h>

 std::mt19937 generator(0); // random number generator with consistent seed for reproducible results.

typedef std::vector<double> vector; // shortcut since we use this datatype a lot
// vectors are useful to allow us to define arrays with arbitrary size

vector randomVector(int dimensions, double low, double high){ // returns a vector of length dimensions with each number randomly uniformly distributed between low and high
	std::uniform_real_distribution<> distribution(low, high); // random numbers between low and high
	vector vec = vector(dimensions);
	for (int i=0; i<dimensions; i++){
		vec[i] = distribution(generator); // generates random number
	}
	return vec;
};

class Particle{ // essentially a container class with public member variables that holds the information relevant to a particle

	public:
		vector position, bestPosition, velocity;
		double loss, bestLoss;

		Particle(int numDimensions, double low, double high){ // particle is initialized with a size (number of dimensions/parameters) and a range for those parameters
			this->position = randomVector(numDimensions, low, high); // initial position is random vector
			this->bestPosition = this->position; // initial best position is a (copy of) the initial position
			this->velocity = randomVector(numDimensions, 0, 0); // most PSOs initialize particle velocity to zero
			this->loss = DBL_MAX; // assume worst possible loss originally (since the loss of a particle is determined externally)
			this->bestLoss = DBL_MAX; 
		}

};

typedef std::vector<Particle> pso; // another type declaration to make things clear

class Swarm{ // a class that contains a collection of particles, a global best particles, and methods for updating the positions and velocities of particles in the swarm based on a function to optimize
	private:
		double rastrigin(vector v){ // the loss function to minimize. A highly multimodal function with a minimum at the zero vector.
			double result = v.size();
			for (double x : v){
				result += x*x - cos(2*3.14159*x);
			}
			return result;
		}

		void updateSwarmPersistent(double &inertia, double &cognitive, double &social){ // a method that takes pointers to update parameters (so that after updating the swarm is called, modifications to parameters persists)
			Particle &gBest = this->globalBest; // the global best particle
			vector &globalBestPosition = gBest.bestPosition; // the position of the global best particle

			for (Particle &p : this->swarm){ // for each particle in the swarm
				double oldLoss = p.loss; // hold on to the old loss temporarily
				vector &position = p.position;
				vector &bestPosition = p.bestPosition;
				vector &velocity = p.velocity;

				vector random1 = randomVector(position.size(), 0, 1); // random vectors used to take random amount of vector pointing from current position to personal best and global best positions
				vector random2 = randomVector(position.size(), 0, 1);

				for (int i=0; i<position.size(); i++){ // for each dimension in the particles position and velocity vectors

					velocity[i] *= inertia; // scale the previous velocity by the inertia
					velocity[i] += cognitive*random1[i]*(bestPosition[i] - position[i]); // take a random amount of the vector pointing from current position to personal best position and scale it by cognitive factor
					velocity[i] += social*random2[i]*(globalBestPosition[i] - position[i]);// take a random amount of the vector pointing from current position to global best position and scale it by social factor
					position[i] += velocity[i]; // update position using velocity vector
				}

				double loss = rastrigin(position); // calculate the new loss of the particle
				p.loss = loss; // update the particle's loss to match

				if (loss < p.bestLoss){ // see if we need to update our personal best
					p.bestLoss = p.loss; // if so, update the best loss and best positions to match
					p.bestPosition = p.position;

					if (loss < gBest.bestLoss){ // we only need to check if we beat the global best if we update the personal best
						gBest.bestLoss = loss; // if we beat the global best's loss, then update the global best loss and global best positions to match
						gBest.bestPosition = p.position; 
					}
				}	
			}
		}

	public:
		Particle globalBest = Particle(0,0,0); // The global best particle in the swarm (C++ requires Particle to be initialized since it doesn't have a default constructor)
		pso swarm; // Collection of Particle objects

		Swarm(int numParticles, int numDimensions, double low, double high){ // Creates a collection of numParticles particles, each with dimensions numDimensions, whose initial position components are between low and high

			Particle tempParticleForInitializingVector = Particle(0,0,0); // temporary particle object used to initialize the swarm collection
			this->swarm = pso(numParticles, tempParticleForInitializingVector); // create the swarm collection by mkaing numParticles copies of tempParticleForInitializingVector

			double bestLoss = DBL_MAX; // assume that the best loss of all the particles right now is as bad as possible

			for (Particle &p : this->swarm){ // for each particle in the swarm
				p = Particle(numDimensions, low, high); // make it a new particle
				double loss = rastrigin(p.position); // calculate its loss
				p.loss = loss; // set its loss
				p.bestLoss = loss; // and best loss to this position

				if (loss < bestLoss){ // check to see if we are beating the global best loss
					bestLoss = loss; // update the bestLoss to beat with this particles loss
					globalBest.bestPosition = vector(p.position); // copy constructor, update the global best's position with this particle's position
					globalBest.bestLoss = loss; // update the global best's loss with this particle's loss
				}

			}
		}

		void iterateSwarmPersistent(int behaviour, int generations, double &inertia, double &cognitive, double &social){ // method that updates the entire swarm generations times, using the inertia, cognitive and social parameters passed in. it takes an integer behaviour between 0 and 3 inclusive that determines how these parameters change over time.
			double gamma = 1.0 - 1.0/generations; // a decay factor that scales inversely with the number of proportions. Note that for repeated scaling of parameters through multiplication (for example inertia decay), after n generations, we would have inertia*(1-1/n)^n which tends to inertia/e for n approaching infinity, e being the base of the natural exponent

			for (int g=0; g<generations; g++){ // iterate for each generation
				switch (behaviour){ // based on the behaviour, modify parameters in different ways
					// case 0 omitted since it does nothing
					case 1: // inertia decay (inertia decreases over time)
						inertia *= gamma;
						break;
					case 2: // exploration-exploitation switch (over time, social factor dominates and cognitive factor decays)
						social /= gamma;
						cognitive *= gamma;
						break;
					case 3: // case 1 and 2 combined
						inertia *= gamma;
						social /= gamma;
						cognitive *= gamma;
						break;
				}

				this->updateSwarmPersistent(inertia, cognitive, social); // calling updateswarmpersistent ensures that after updating the swarm, inertia, cognitive and social maintain their modifications
			}
		}

		void iterateSwarm(int behaviour, int generations, double inertia, double cognitive, double social){ // just a method for calling with straight numbers without worrying about having variables for each of these
			iterateSwarmPersistent(behaviour, generations, inertia, cognitive, social);
		}

		void printLosses(){ // utility function to check the best losses of every particle in the swarm
			for (Particle &p : this->swarm){
				std::cout << p.bestLoss << "\n";
			}
		}

};




typedef std::vector<Swarm> superswarm; // allows us to have a collection of swarms

void bcastGather(int iterations, // method that generates numSwarms swarms, each containing numParticles particles with numDimensions dimensions, initialized with components between low and high. runs iterations runs of generations updates using inertia, cognitive and social parameters.
		int numSwarms, int numParticles, int numDimensions,\
		double low, double high,\
		int generations, double inertia, double cognitive, double social){

	int communicationSize; // number of processes
  	int processRank; // rank of current process

  	MPI_Init(NULL, NULL); 

 	MPI_Comm_size(MPI_COMM_WORLD, &communicationSize); // set communicationSize with the appropriate number of processes

 	MPI_Comm_rank(MPI_COMM_WORLD, &processRank); // set processRank with the appropriate process number

	Swarm tempSwarmForInitializingVector = Swarm(0,0,0,0); // just used to set up superswarm since there is no default swarm constructor
	superswarm super = superswarm(numSwarms, tempSwarmForInitializingVector); // set up superswarm with numSwarms copies of tempSwarmForInitializingVector
	for (Swarm &s : super){ // then for each swarm in the superswarm
		s = Swarm(numParticles, numDimensions, low, high); // set that swarm to a new swarm object (so that they are all unique)
	}


	// After each iteration (generation runs of updates), we collect the global best losses and positions from each swarm
	// Then we decide which is best and update every swarm to have the same global best particle
	// The point of having two is to avoid recalculating the loss on position vectors if we already have it

	double lossesBuffer[communicationSize]; // used to collect the loss of the global bests from each swarm
	double positionsBuffer[communicationSize*numDimensions];// used to collect the position of the global bests from each swarm

	Particle &gBest = super[0].globalBest; // we have a reference to the global best out of all the particles (the superswarms global best in a way) and we initialize it to the first swarm's global best

	for (int i=0; i<iterations; i++){ // for each iteration

		// within a process, we can have multiple swarms run sequentially with the same behaviour, and find out of them the best.

		for (Swarm &s : super){ // for each swarm
			s.iterateSwarm(processRank%4, generations, inertia, cognitive, social); // update the swarm using a behaviour defined the the process rank
			double loss = s.globalBest.bestLoss; // get the loss of the swarms global best

			if (loss < gBest.bestLoss){ // if this swarms global best loss is better than the superswarms global best
				gBest = s.globalBest; // update our reference to point to this swarms global best
			}
		
		}

		double bestLoss = gBest.bestLoss; // a buffer to hold the global best loss out of all of the swarms across all processes
		double bestPosition[numDimensions]; // same but for global best position 

		for (int j=0; j<numDimensions; j++){ // mismatch in types: vector vs double[] (the latter being needed to transmit data using OpenMPI)
			bestPosition[j] = gBest.bestPosition[j];
		}

		// Rather than gathering with or without recieving buffers based on process, just have everyone send and recieve to their respective buffers, but have only process 0 do the work to determine the global best out of all processses

		MPI_Gather(&bestLoss, 1, MPI_DOUBLE, &lossesBuffer, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); // let everyone send their best loss to the best loss buffer
		MPI_Gather(&bestPosition, numDimensions, MPI_DOUBLE, &positionsBuffer, numDimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD); // and let everyone send their best position to the best position buffer

		if (processRank == 0){ // if we are the parent process, we are responsible for determining the global best position out of everyone

			// since processes fill buffers based on their process (eg; process 0 sends best loss to position 0 of lossesBuffer) we can use this to determine the chunk of the positionsBuffer containing the best position corresponding to that loss
			double minLoss = DBL_MAX; // assume the best loss is a bad as possible
			int argmin = -1; // we can assume the argument of that loss is -1 since it doesnt correspond to a real position
			for (int l = 0; l<communicationSize; l++){ // for each loss from each process
				if (lossesBuffer[l] < minLoss){ // if this is the best loss
					minLoss = lossesBuffer[l]; // update the best loss
					argmin = l; // and update the position of this best loss
				}
			}

			bestLoss = lossesBuffer[argmin]; // fill up bestLoss buffer with this best loss

			for (int j=0; j<numDimensions; j++){ // fill up bestPosition buffer with the correct position from the positions buffer
				bestPosition[j] = positionsBuffer[argmin*numDimensions + j]; // each position has a chunk size of numDimensions, so the position from process i starts at i*numDimensions
			}
		}

		// Everyone calls MPI_Bcast, only process 0 sends information out, everyone else fills their buffers with this info

		MPI_Bcast(&bestLoss, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&bestPosition, numDimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		vector bestPos = vector(bestPosition, bestPosition+numDimensions); // create a vector from double[] (again, since OpenMPI uses double[] for buffer)
		gBest.bestLoss = bestLoss; // update the global best loss
		gBest.bestPosition = bestPos; // and position so that every processor has the same global best

		for (Swarm &s : super){ // share this global best among all the swarms running in this process
			s.globalBest = Particle(gBest); 
		}

		std::printf("global best loss: %f\n", gBest.bestLoss); // keep track of the best loss

	}	

	MPI_Finalize();
}

int main(int argc, char **argv){

	// compile with mpicc -lc++ -g -o PSO-MPI PSO.cpp

	int numSwarms = std::stoi(argv[1]); // when running from CLI you run time mpiexec -n N PSO-MPI K where K is the number of swarms to run per processor (eg: time mpiexec -n 2 PSO-MPI 8)

	// fairly standard for testing

	int iterations = 10;
	int numParticles = 20;
	int numDimensions = 100;
	double low = -5.12;
	double high = 5.12;
	int generations = 250;
	double inertia = 0.7;
	double cognitive = 1.5;
	double social = 1.5;

	bcastGather(iterations, numSwarms, numParticles, numDimensions, low, high, generations, inertia, cognitive, social);

}


