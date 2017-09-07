
public class Environment {
	
	
	public char map[][] = {
			{'S','F','F','F'},	
			{'F','H','F','H'},	
			{'F','F','F','H'},	
			{'H','F','F','G'},	
	};
	
	public static final int IDobservation = 0, IDreward = 1, IDdone = 2, IDinfo = 3;
	double observation = 0;
	double reward = 0.0;
	double done = 0.0;
	double info = 0.0;
	
	int posx = 0, posy = 0;
	
	String lastAction = "(none)";
	
	public static final int UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3;
	
	public Environment(){
		
	}
	
	public void reset(){
		observation = 0;
		reward = 0.0;
		done = 0.0;
		info = 0.0;
		posx = 0;
		posy = 0;
	}
	
	public double[] step(int action){
		if(action == UP && posy > 0){
			posy--;
			lastAction = "(up)";
		}
		if(action == DOWN && posy < map.length){
			posy++;
			lastAction = "(down)";
		}
		if(action == LEFT && posx > 0){
			posx--;
			lastAction = "(left)";
		}
		if(action == RIGHT && posx < map[0].length){
			posx++;
			lastAction = "(right)";
		}
		if(map[ posy ][ posx ] == 'H'){
			done = 1.0;
			reward = 0.0;
		}
		if(map[ posy ][ posx ] == 'G'){
			done = 1.0;
			reward = 1.0;
		}
		observation = (posy*map[0].length)+posx;
		
		double ret[] = {observation, reward, done, info};
		return ret;
	}
	
	public void render(){
		for(int y = 0; y < map.length; y++){
			for(int x = 0; x < map[0].length; x++){
				if(x == posx && y == posy){
					System.out.print(">"+map[y][x]+"<");
				}else{
					System.out.print(" "+map[y][x]+" ");
				}
			}
			System.out.println();
		}
		System.out.println(lastAction);
	}

}
