package peersim.MultiSvm;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.net.MalformedURLException;
import java.text.ParseException;
import java.util.HashSet;
import java.util.Map;
import java.util.TreeMap;

	import jnipegasos.JNIPegasosInterface;
import jnipegasos.LearningParameter;
import jnipegasos.PrimalSVMWeights;
import peersim.config.Configuration;
import peersim.core.Cleanable;
import peersim.core.CommonState;
import peersim.core.Fallible;
import peersim.core.Linkable;
import peersim.core.Network;
import peersim.core.Node;
import peersim.core.Protocol;
import weka.classifiers.functions.SGD;
import weka.core.*;
import weka.filters.Filter;


public class MyNode implements Node {
	
		// The gradient at the node
		public double[][] local_loss_sgd;
		public double[][] local_sgd;
		
		// The protocols on current node
		protected Protocol[] protocol = null;
		
		public static HashSet ClassSet= new HashSet();
		
		public double frobenius_norm= 0.0;

		//public String[] classArray= new String[3];
		/**
	 * New config option added to get the resourcepath where the resource file
	 * should be generated. Resource files are named <ID> in resourcepath.
	 * @config
	 */
	private static final String PAR_PATH = "resourcepath";
	
	  private static final String PAR_SIZE = "network.size";


	/** used to generate unique IDs */
	private static long counter = -1;

	
	//protected int lid;
	//the weight vector at node
	public double[][] wtVec;
	
	public int num_class; //total no. of classes
	
	public int num_Att;
	public static  int c;
	
	public int count=0;
	private int index; // current index of any node

	/**
	 * The fail state of the node.
	 */
	protected int failstate = Fallible.OK;

	
	private long ID; //The ID of the node.

	/**
	 * The prefix for the resources file. All the resources file will be in prefix 
	 * directory. later it should be taken from configuration file.
	 */
	/** Learning parameter */
	protected int N;
	public int max_class=10;
	protected double lambda=0.01;
	/** Learning rate */
	protected double alpha=0.02;
	private String resourcepath;
	 
	public Instances traindataset; // The training dataset
	
	public MyNode(String prefix) 
	{
		String[] names = Configuration.getNames(PAR_PROT); // protocol
		
		
		resourcepath = (String)Configuration.getString(prefix + "." + PAR_PATH);
		//N= Configuration.getInt(prefix + "."+PAR_SIZE);
		System.out.println("Data is saved in: " + resourcepath + "\n"); //
		CommonState.setNode(this); //sets current node
		protocol = new Protocol[names.length];
		setID(nextID());
		//ID = nextID(); //node ID starts from zero
		setProtocol(new Protocol[names.length]);
		for (int i=0; i < names.length; i++) {
			CommonState.setPid(i); //sets protocol identifier
			Protocol p = (Protocol) 
					Configuration.getInstance(names[i]);
			getProtocol()[i] = p;
									
		}
	}

	private long nextID() {
		// TODO Auto-generated method stub
		counter=counter+1;
		
		return counter;
	}

	@Override
	public int getFailState() {
		// TODO Auto-generated method stub
		return failstate;
	}

	@Override
	public boolean isUp() {
		// TODO Auto-generated method stub
		return failstate==OK;
	}

	@Override
	public void setFailState(int failState) 
	{
			// after a node is dead, all operations on it are errors by definition
			if(failstate==DEAD && failState!=DEAD) throw new IllegalStateException(
					"Cannot change fail state: node is already DEAD");
			switch(failState)
			{
			case OK:
				failstate=OK;
				break;
			case DEAD:
				//protocol = null;
				index = -1;
				failstate = DEAD;
				for(int i=0;i<protocol.length;++i)
					if(protocol[i] instanceof Cleanable)
						((Cleanable)protocol[i]).onKill();
				break;
			case DOWN:
				failstate = DOWN;
				break;
			default:
				throw new IllegalArgumentException(
						"failState="+failState);
			}
			
		}
		
	

	@Override
	public long getID() {
		// TODO Auto-generated method stub
		return ID;
	}

	@Override
	public int getIndex() {
		// TODO Auto-generated method stub
		return index;
	}

	@Override
	public Protocol getProtocol(int i) {
		// TODO Auto-generated method stub
		return protocol[i];
	}

	@Override
	public int protocolSize() {
		// TODO Auto-generated method stub
		return getProtocol().length;
	}

	@Override
	public void setIndex(int index) {
		// TODO Auto-generated method stub
		this.index=index;
	}
 
	
	public void setID(long nextID) {
		// TODO Auto-generated method stub
		ID=nextID;
		
	}
	
	private Protocol[] getProtocol() {
		return protocol;
	}
	private void setProtocol(Protocol[] protocol) {
		this.protocol = protocol;
		
	}
	
	public Object clone() 
	{
		MyNode result = null;
		try { 
			result=(MyNode)super.clone();
		}
		catch( CloneNotSupportedException e ) {
			e.printStackTrace();
			} 
		result.setProtocol(new Protocol[protocol.length]);
	
		CommonState.setNode(result);
		result.setID(nextID());
			
		for(int i=0;i<getProtocol().length;i++)
		{
			CommonState.setPid(i);
			result.getProtocol()[i] = (Protocol)getProtocol()[i].clone(); //
		}

//read the data into node
try{
	String traindataset = resourcepath + "/" + "iris_dataset_" + result.getID() + ".arff";
	//Instances data= traindataset.getDataSet();
	FileReader reader = new FileReader(traindataset);
    Instances data = new Instances (reader);
    
    // printing the number of attributes
     num_Att = data.numAttributes()-1;
    System.out.println("Number of Attributes at node "+result.getID()+" "+num_Att+ "\n");
    //total number of instances
    int num_Instance=data.numInstances();
    System.out.println("Number of instances at node "+result.getID()+" "+num_Instance+ "\n");
    	     
    //set the last attribute to be the class attribute
    int label=data.numAttributes();
   data.setClassIndex(label-1); 
   // System.out.println("\n"+label);
    
    
    result.traindataset=data;
    //retrieve the different class at each node
    int N = data.numInstances();	// data size at each node
    for(int i=0;i<N;i++)    // Adding unique classes to Hashset
    {
    	double y= data.instance(i).value(label-1);
    	//System.out.println("\n"+y);
    	ClassSet.add(y);
    	
    } 
   
    
    //System.out.println("The Total number of known class:"+c);
    num_Att=data.numAttributes()-1;
    
    //Initialize the weight vector dxc dimension where d is the num ber of attributes and c is the total number of classes
    wtVec = new double[num_Att][max_class];
	    
    for(int j=0; j<num_Att; j++)
    { 
    	for(int k=0;k<max_class;k++)
        {
    	wtVec[j][k]=1;
    	 
    }
    }

	 //MyNode n =(MyNode) node;
	 num_Att=data.numAttributes()-1;
	 //num_class=ClassSet.size();
    local_sgd = new double[num_Att][max_class];
    for (int row = 0; row < num_Att; row ++)
    {
   	    for (int col = 0; col < max_class; col++)
   	    {
   	        local_sgd[row][col] = 0.0;
   	        		
   	    }    	
    }
    
    //System.out.println(num_class);
  //Building the local model and calculating the local subgradient at each node
    local_loss_sgd=new double[num_Att][max_class];// the gradient matrix
    for (int row = 0; row < num_Att; row ++)
    {
   	    for (int col = 0; col < max_class; col++)
   	    {
   	        local_loss_sgd[row][col] = 0.0;
   	        		
   	    }    	
    }
    

	    double y;
	    double r = 0.0;
		//
		//int N1 = data.size();
		for(int i=0;i<N;i++)
		{
			double dot_prod=0.0;
			
			double [] wx= new double[max_class];
			int x_size= data.numAttributes()-1;
			y=data.instance(i).classValue();
			System.out.println("the value of label for input:"+i+"is"+y);
			for(int c1=0;c1<max_class;c1++)
			{
			
			  for (int xiter = 0; xiter < x_size; xiter++) 
			    { //inner dot product loop
				// input value
				double xval = data.instance(i).value(xiter);
				
				// wtvector value
					double wval = wtVec[xiter][c1];
					dot_prod =dot_prod +( xval * wval);
				}// dot product loop end
		     wx[c1]=dot_prod;
		     dot_prod=0.0;
		     System.out.println("The dot product at "+result.getID()+"for input:"+ i+"is "+wx[c1]+" \n ");
          }       
			double max=0.0;
			for(int z=0;z<max_class;z++)
			{
				if(max<wx[z])
				{
					r=z+1;
					max=wx[z];
				}
			}
			System.out.println(r);
			//for each training example compute the gradient
			double[][] sgd=new double[x_size][max_class];
			
			for(int c1=0;c1<max_class;c1++)
			{
			  for (int xiter = 0; xiter < x_size; xiter++) 
			     { //inner loop input value
					double xval =data.instance(i).value(xiter);
			        if(c1==r-1)
				    {
					    sgd[xiter][c1]=xval;
				    }
			        else if(c1==y-1)
			        {
			        	sgd[xiter][c1]=-xval;
			        }
			        else
			        	{sgd[xiter][c1]=0.0;}
			        //System.out.println("sgd is :"+sgd[xiter][c1]+"\n");
			        
			        local_loss_sgd[xiter][c1]=local_loss_sgd[xiter][c1]+sgd[xiter][c1];
			        //System.out.println("local loss sgd: "+local_loss_sgd[xiter][c1]+"\n");
			      }
			}
			
			
       }
		for (int row = 0; row < num_Att; row ++)
	     {
	    	    for (int col = 0; col < max_class; col++)
	    	    {
		            local_loss_sgd[row][col]=(local_loss_sgd[row][col]/N);
		            wtVec[row][col]=(wtVec[row][col])*lambda;
                }
        }
		for (int row = 0; row < num_Att; row ++)
	     {
	    	    for (int col = 0; col < max_class; col++)
	    	    {
	    	    	local_sgd[row][col]=wtVec[row][col]+ local_loss_sgd[row][col];
	    	    	System.out.println("summation of local loss sgd and p(w): "+local_sgd[row][col]);
	    	    }
	     }
		double [][] newval=new double[num_Att][max_class];
		for (int row = 0; row < num_Att; row ++)
	     {
	    	    for (int col = 0; col < max_class; col++)
	    	    {
	    	    	newval[row][col]=0.0;
	    	    }
	    	    }
		
		// calculating updated weight vector which is w(new)=w(old)-(alpha*(local_sgd))
		for (int row = 0; row < num_Att; row ++)
	     {
	    	    for (int col = 0; col <max_class; col++)
	    	    {
	    	    	newval[row][col]=alpha*(local_sgd[row][col]);
	    	    	//System.out.println(newval[row][col]);
	    	    }
	    }
		for (int row = 0; row < num_Att; row ++)
	     {
	    	    for (int col = 0; col < max_class; col++)
	    	    {
	    	    	wtVec[row][col]=wtVec[row][col]-newval[row][col];
	    	    	System.out.println(wtVec[row][col]);
	    	    }
	     }
//		for (int row = 0; row < num_Att; row ++)
//	     {
//			for(int col=0;col< max_class;col++)
//			{
//			   frobenius_norm=frobenius_norm+Math.pow((wtVec[row][col]),2);
//					  
//            }
//		}
		//System.out.println("The square root"+Math.sqrt(frobenius_norm));
		
  }
	

      
    
    
     
	

catch(Exception e)
{
	e.printStackTrace();
	}

System.out.println("created node with ID: " + result.getID()+ "\n");
//count++;
//if(count==N-1)
//{
//call(result);	
//}

	//System.out.println("Total No. of class"+ClassSet.size());
return result;

}
	

	public static int write_class_size(){
	  c= ClassSet.size();
	//System.out.println("\n"+ClassSet.size());
	return c;
	}


	
}
	


	
	
	


