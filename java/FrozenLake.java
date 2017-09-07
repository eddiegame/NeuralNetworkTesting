import java.util.Random;
import java.util.Scanner;

public class FrozenLake {

	Random random = new Random();
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		//new Player();
		new FrozenLake();
	}
	
	NeuralNetwork nn = new NeuralNetwork();
	
	public FrozenLake(){
		int Inputlength = 16;
		int amountNodesHL = Inputlength+1;
		int Outputlength = 4;
		System.out.println("initializing weights and nodes matrix..");
		nn.init(Inputlength, amountNodesHL, Outputlength);
		
		
		
		Scanner s = new Scanner(System.in);
		Environment env = new Environment();
		int observation = random.nextInt(4);
		while(s.nextLine()!="!"){
			int action = nn.Qvalue(observation);
			double[] info = env.step(action);
			NeuralNetwork.print(info);
			if(info[Environment.IDdone] == 1.0){
				env.reset();
			}
			observation = (int) info[Environment.IDobservation];
			env.render();
		}
		
		s.close();
		
		
		
	}
	

}
