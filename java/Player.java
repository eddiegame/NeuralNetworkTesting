import java.awt.Dimension;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

import javax.swing.JFrame;

public class Player extends JFrame implements KeyListener{
	
	Environment env = new Environment();
	
	public Player(){
		this.addKeyListener(this);
		this.pack();
		this.setMinimumSize(new Dimension(100,100));
		this.setVisible(true);
	}

	@Override
	public void keyPressed(KeyEvent e) {
		if(e.getKeyCode() == KeyEvent.VK_UP){
			NeuralNetwork.print(env.step(Environment.UP));
		}
		if(e.getKeyCode() == KeyEvent.VK_DOWN){
			NeuralNetwork.print(env.step(Environment.DOWN));	
		}
		if(e.getKeyCode() == KeyEvent.VK_LEFT){
			NeuralNetwork.print(env.step(Environment.LEFT));
		}
		if(e.getKeyCode() == KeyEvent.VK_RIGHT){
			NeuralNetwork.print(env.step(Environment.RIGHT));
		}
		if(e.getKeyCode() == KeyEvent.VK_R){
			env.reset();
		}
		if(env.done == 1.0){
			env.reset();
		}
		env.render();
	}

	@Override
	public void keyReleased(KeyEvent arg0) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void keyTyped(KeyEvent arg0) {
		// TODO Auto-generated method stub
		
	}

}
