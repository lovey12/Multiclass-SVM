package peersim.multisvm;

import java.io.FileReader;
import java.util.HashSet;
import java.util.Set;

import peersim.config.Configuration;
import peersim.core.Cleanable;
import peersim.core.CommonState;
import peersim.core.Fallible;
import peersim.core.Node;
import peersim.core.Protocol;
import weka.core.Instances;



public class MultiSVMCustomNode implements Node{
	

    // The gradient at the node
    public double[][] local_loss_sgd;

    public double[][] local_sgd;

    // The protocols on current node
    protected Protocol[] protocol = null;

    public static Set<Double> ClassSet = new HashSet<Double>();

    public double frobenius_norm = 0.0;

    public double norm = 0.0;

    // public String[] classArray= new String[3];
    /**
     * New config option added to get the resourcepath where the resource file
     * should be generated. Resource files are named <ID> in resourcepath.
     * 
     * @config
     */
    private static final String PAR_PATH = "resourcepath";

    private static final String PAR_SIZE = "network.size";

    /** used to generate unique IDs */
    private static long counter = -1;

    // protected int lid;
    // the weight vector at node
    public double[][] wtVec;

    public int num_class; // total no. of classes

    public int num_Att;

    public static int c;

    public int count = 0;

    private int index; // current index of any node

    /**
     * The fail state of the node.
     */
    protected int failstate = Fallible.OK;

    private long ID; // The ID of the node.

    /**
     * The prefix for the resources file. All the resources file will be in
     * prefix directory. later it should be taken from configuration file.
     */
    /** Learning parameter */
    protected int N;

    public int max_class = 10;

    protected double lambda = 0.01;

    /** Learning rate */
    protected double alpha = 0.02;

    private String resourcepath;

    public Instances traindataset; // The training dataset

    private long nextID()
    {

        // TODO Auto-generated method stub
        

        return counter++;
    }
    
	public int getFailState() {
		// TODO Auto-generated method stub
		return failstate;
	}

	public void setFailState(int failState) {
		// TODO Auto-generated method stub
		// after a node is dead, all operations on it are errors by definition
        if (failstate == DEAD && failState != DEAD)
            throw new IllegalStateException(
                "Cannot change fail state: node is already DEAD");
        switch (failState)
        {
        case OK:
            failstate = OK;
            break;
        case DEAD:
            // protocol = null;
            index = -1;
            failstate = DEAD;
            for (int i = 0; i < protocol.length; ++i)
                if (protocol[i] instanceof Cleanable)
                    ((Cleanable) protocol[i]).onKill();
            break;
        case DOWN:
            failstate = DOWN;
            break;
        default:
            throw new IllegalArgumentException(
                "failState=" + failState);
        }
	}

	public boolean isUp() {
		// TODO Auto-generated method stub
		return failstate == OK;
	}

	public Protocol getProtocol(int i) {
		// TODO Auto-generated method stub
		return protocol[i];
	}

	public int protocolSize() {
		// TODO Auto-generated method stub
		return getProtocol().length;
	}

	public void setIndex(int index) {
		// TODO Auto-generated method stub
		this.index = index;
	}

	public int getIndex() {
		// TODO Auto-generated method stub
		return index;
	}
	 public void setID(long nextID)
	    {

	        ID = nextID;

	    }
	public long getID() {
		// TODO Auto-generated method stub
		return ID;
	}
	 private Protocol[] getProtocol()
	    {

	        return protocol;
	    }

	    private void setProtocol(Protocol[] protocol)
	    {

	        this.protocol = protocol;

	    }
	    
	   public MultiSVMCustomNode(String prefix)
	    {

	        String[] names = Configuration.getNames(PAR_PROT); // protocol

	        resourcepath = (String) Configuration.getString(prefix + "." + PAR_PATH);
	        // N= Configuration.getInt(prefix + "."+PAR_SIZE);
	        System.out.println("Data is saved in: " + resourcepath + "\n"); //
	        CommonState.setNode(this); // sets current node
	        protocol = new Protocol[names.length];
	        setID(nextID());
	        // ID = nextID(); //node ID starts from zero
	        setProtocol(new Protocol[names.length]);
	        for (int i = 0; i < names.length; i++) {
	            CommonState.setPid(i); // sets protocol identifier
	            Protocol p = (Protocol)
	                Configuration.getInstance(names[i]);
	            getProtocol()[i] = p;

	        }
	    }
	
	
	   public Object clone()
	    {

	        MultiSVMCustomNode result = null;
	        try {
	            result = (MultiSVMCustomNode) super.clone();
	        }
	        catch (CloneNotSupportedException e) {
	            e.printStackTrace();
	        }
	        result.setProtocol(new Protocol[getProtocol().length]);

	        CommonState.setNode(result);
	        result.setID(nextID());

	        for (int i = 0; i < getProtocol().length; i++)
	        {
	            CommonState.setPid(i);
	            result.getProtocol()[i] = (Protocol) getProtocol()[i].clone(); //
	        }

	        // read the data into node
	        try {
	            String traindataset = resourcepath + "/" + "waveform_dataset_"
	            		+ "" + result.getID() + ".arff";
	            // Instances data= traindataset.getDataSet();
	            FileReader reader = new FileReader(traindataset);
	            Instances data = new Instances(reader);

	            // printing the number of attributes
	            num_Att = data.numAttributes() - 1;
	            System.out.println("Number of Attributes at node " + result.getID() + " " + num_Att + "\n");
	            // total number of instances
	            int num_Instance = data.numInstances();
	            System.out.println("Number of instances at node " + result.getID() + " " + num_Instance + "\n");

	            // set the last attribute to be the class attribute
	            int label = data.numAttributes();
	            data.setClassIndex(label - 1);
	            // System.out.println("\n"+label);

	            result.traindataset = data;
	            // retrieve the different class at each node
	            int N = data.numInstances(); // data size at each node
	            for (int i = 0; i < N; i++) // Adding unique classes to Hashset
	            {
	                double y = data.instance(i).value(label - 1);
	                // System.out.println("\n"+y);
	                ClassSet.add(y);

	            }

	            // System.out.println("The Total number of known class:"+c);
	            num_Att = data.numAttributes() - 1;

	            // Initialize the weight vector dxc dimension where d is the num ber
	            // of attributes and c is the total number of classes
	            wtVec = new double[num_Att][max_class];

	            for (int j = 0; j < num_Att; j++)
	            {
	                for (int k = 0; k < max_class; k++)
	                {
	                    wtVec[j][k] = 1;

	                }
	            }

	            // MyNode n =(MyNode) node;
	            num_Att = data.numAttributes() - 1;
	        }
	
	
	        catch (Exception e)
	        {
	            e.printStackTrace();
	        }

	        System.out.println("created node with ID: " + result.getID() + "\n");
	        // count++;
	        // if(count==N-1)

	        return result;

	    }
	   
	   public static int write_class_size()
	    {

	        c = ClassSet.size();
	        // System.out.println("\n"+ClassSet.size());
	        return c;
	    }


}
