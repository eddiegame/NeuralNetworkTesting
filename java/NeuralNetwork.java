import java.util.Random;

public class NeuralNetwork {
	
	double[][][] weights;
	double[][] nodes;
	double[][] absnodes;
	
	Random random = new Random();
	
	double lear_rate   = 0.1;
	double muta_rate   = 0.3;
	double gamma       = 0.9;
	
	public void init(int Inputlength, int amountNodesHL, int Outputlength){
		double[][][] 	weights = {		
				zeros(amountNodesHL,Inputlength),
				zeros(amountNodesHL,amountNodesHL),
				zeros(Outputlength,amountNodesHL )};
		double[][] 		nodes = {		zeros(Inputlength),
						zeros(amountNodesHL),
						zeros(amountNodesHL),
						zeros(Outputlength)};
		double[][] 		absnodes = nodes;
		
		// init random Weights
				for (int layer = 0; layer < 3; layer++){
				    for (int x = 0; x < weights[layer].length; x++){
				        for (int y = 0; y < weights[layer][x].length; y++){
				            weights[layer][x][y] = random.nextDouble();
						}
					}
				}
	}

	public int Qvalue(int observation) {
		nodes[0][observation]=1;
		
		int layer = 1;
	    while(layer < nodes.length){
	        // für jeden Node im Layer layer
	        int node = 0;
	        while(node < nodes[layer].length){
	            // erhöhe den wert in nodes[layer][node] um jedes Produkt der Weights und deren Nodes von denen sie kommen
	            int weight = 0;
	            while(weight < weights[layer-1][node].length){
	                // aktueller node += weight des Pfades (layer-1) * node im vorherigen Layer von dem der Pfad kommt
	                absnodes[layer][node]+=(weights[layer-1][node][weight]*nodes[layer-1][weight]);
	                nodes[layer][node] = sigmoid(absnodes[layer][node]);
	                weight+=1;
	            }
	            node+=1;
	        }
	        layer+=1;
	    }
		//print(nodes);
	    int action = 0;
		double max = nodes[3][0];
		for(int i = 0; i < nodes[3].length; i++){
			if(nodes[3][i] > max){
				action = i;
			}
		}
		return action;
	}
	
	public void learn(double observation, double reward){
		int action = Qvalue((int) observation);
		
		double[][][] learnweights = weights;
		double[][] learnnodes = nodes;
		
		learnnodes[3][action] = 1;
		
	}
	
	public static void print(double[] d){
		System.out.print("[ ");
		for(int i = 0; i < d.length; i++){
			System.out.print(d[i]);
			if(i+1 < d.length){
				System.out.print(", ");
			}
			
		}
		System.out.println("]");
	}
	
	public static void print(double[][] d){
		for(double[] doub : d){
			print(doub);
		}
	}
	
	public static void print(double[][][] d){
		for(double[][] doub : d){
			print(doub);
		}
	}
	
	public static double sigmoid(double x){
		return 1 / (1 + Math.exp(-x));
	}


	public double[][][] zeros(int sizex, int sizey, int sizez){
		double d[][][] = new double[sizex][sizey][sizez];
		for(int x = 0; x < sizex; x++){
			for(int y = 0; y < sizey; y++){
				for(int z= 0; y < sizez; z++){
					d[x][y][z] = 0.0;
				}
			}
		}
		return d;
	}
	
	public double[][] zeros(int sizex, int sizey){
		double d[][] = new double[sizex][sizey];
		for(int x = 0; x < sizex; x++){
			for(int y = 0; y < sizey; y++){
				d[x][y] = 0.0;
			}
		}
		return d;
	}
	
	public double[] zeros(int sizex){
		double d[] = new double[sizex];
		for(int x = 0; x < sizex; x++){
			d[x] = 0.0;
		}
		return d;
	}
	
}
