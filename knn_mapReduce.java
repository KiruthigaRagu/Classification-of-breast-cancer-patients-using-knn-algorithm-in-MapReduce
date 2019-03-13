import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.File;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.TreeMap;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class KnnPattern
{
	
	// WritableComparable class for a paired Double and String (distance and model)
	// This is a custom class for MapReduce to pass a double and a String through context
	// as one serializable object.
	// This example only implements the minimum required methods to make this job run. To be
	// deployed robustly is should include ToString(), hashCode(), WritableComparable interface
	// if this object was intended to be used as a key etc.
		public static class DoubleString implements WritableComparable<DoubleString>
		{
			private Double distance = 0.0;
			private String model = null;

			public void set(Double lhs, String rhs)
			{
				distance = lhs;
				model = rhs;
			}
			
			public Double getDistance()
			{
				return distance;
			}
			
			public String getModel()
			{
				return model;
			}
			
			@Override
			public void readFields(DataInput in) throws IOException
			{
				distance = in.readDouble();
				model = in.readUTF();
			}
			
			@Override
			public void write(DataOutput out) throws IOException
			{
				out.writeDouble(distance);
				out.writeUTF(model);
			}
			
			@Override
			public int compareTo(DoubleString o)
			{
				return (this.model).compareTo(o.model);
			}
		}
	
	// The mapper class accepts an object and text (row identifier and row contents) and outputs
	// two MapReduce Writable classes, NullWritable and DoubleString (defined earlier)
	public static class KnnMapper extends Mapper<Object, Text, NullWritable, DoubleString>
	{
		DoubleString distanceAndModel = new DoubleString();
		TreeMap<Double, String> KnnMap = new TreeMap<Double, String>();
		
		// Declaring some variables which will be used throughout the mapper
		int K;
	    
		double normalisedScode;
		double normalisedSclump;
		double normalisedScellsize;
		double normalisedScellshape;
		double normalisedSmarginal;
		double normalisedSsinglecellsize;
		double normalisedSbarenuclei;
		double normalisedSchromatin;
		double normalisedSnucleoli;
		double normalisedSmitosis;
		
		
		// The known ranges of the dataset, which can be hardcoded in for the purposes of this example
		double mincode=61634;
		double minclump=1;
		double mincellsize=1;
		double mincellshape=1;
		double minmarginal=1;
		double minsinglecellsize=1;
		double minbarenuclei=1;
		double minchromatin=1;
		double minnucleoli=1;
		double minmitosis=1;
		
		double maxcode=13454352;
		double maxclump=10;
		double maxcellsize=10;
		double maxcellshape=10;
		double maxmarginal=10;
		double maxsinglecellsize=10;
		double maxbarenuclei=10;
		double maxchromatin=10;
		double maxnucleoli=10;
		double maxmitosis=10;
			
		// Takes a string and two double values. Converts string to a double and normalises it to
		// a value in the range supplied to reurn a double between 0.0 and 1.0 
		private double normalisedDouble(String n1, double minValue, double maxValue)
		{
			return (Double.parseDouble(n1) - minValue) / (maxValue - minValue);
		}
		
		// Takes two strings and simply compares then to return a double of 0.0 (non-identical) or 1.0 (identical).
		// This provides a way of evaluating a numerical distance between two nominal values.
		private double nominalDistance(String t1, String t2)
		{
			if (t1.equals(t2))
			{
				return 0;
			}
			else
			{
				return 1;
			}
		}
		
		// Takes a double and returns its squared value.
		private double squaredDistance(double n1)
		{
			return Math.pow(n1,2);
		}
		

		// Takes ten pairs of values (three pairs of doubles and two of strings), finds the difference between the members
		// of each pair (using nominalDistance() for strings) and returns the sum of the squared differences as a double.
		private double totalSquaredDistance(double R1, double R2, double R3, double R4, double R5, double R6,double R7,double R8,double R9,double R10,double S1,
				double S2, double S3, double S4, double S5,double S6,double S7,double S8,double S9,double S10)
		{	
			double codedifference=S1-R1;
			double clumpdifference=S2-R2;
			double cellsizedifference=S3-R3;
			double cellshapedifference=S4-R4;
			double marginaldifference=S5-R5;
			double singlecellsizedifference=S6-R6;
			double barenucleidifference=S7-R7;
			double chromatindifference=S8-R8;
			double nucleolidifference=S9-R9;
			double mitosisdifference=S10-R10;
			// The sum of squared distances is used rather than the euclidean distance
			// because taking the square root would not change the order.
			// Status and gender are not squared because they are always 0 or 1.
			return squaredDistance(codedifference) + squaredDistance(clumpdifference) + squaredDistance(cellsizedifference) + squaredDistance(cellshapedifference) + squaredDistance(marginaldifference) + squaredDistance(singlecellsizedifference) + squaredDistance(barenucleidifference) + squaredDistance(chromatindifference) + squaredDistance(nucleolidifference) + squaredDistance(mitosisdifference);
		}

		// The @Override annotation causes the compiler to check if a method is actually being overridden
		// (a warning would be produced in case of a typo or incorrectly matched parameters)
		@Override
		// The setup() method is run once at the start of the mapper and is supplied with MapReduce's
		// context object
		protected void setup(Context context) throws IOException, InterruptedException
		{
			if (context.getCacheFiles() != null && context.getCacheFiles().length > 0)
			{
				// Read parameter file using alias established in main()
				String knnParams = FileUtils.readFileToString(new File("./knnParamFile"));
				StringTokenizer st = new StringTokenizer(knnParams, ",");
		    	
		    	// Using the variables declared earlier, values are assigned to K and to the test dataset, S.
		    	// These values will remain unchanged throughout the mapper
				K = Integer.parseInt(st.nextToken());
				
				normalisedScode=normalisedDouble(st.nextToken(),mincode,maxcode);
					 normalisedSclump=normalisedDouble(st.nextToken(),minclump,maxclump);
					 normalisedScellsize=normalisedDouble(st.nextToken(),mincellsize,maxcellsize);
					normalisedScellshape=normalisedDouble(st.nextToken(),mincellshape,maxcellshape);
					 normalisedSmarginal=normalisedDouble(st.nextToken(),minmarginal,maxmarginal);
					 normalisedSsinglecellsize=normalisedDouble(st.nextToken(),minsinglecellsize,maxsinglecellsize);
					 normalisedSbarenuclei=normalisedDouble(st.nextToken(),minbarenuclei,maxbarenuclei);
					 normalisedSchromatin=normalisedDouble(st.nextToken(),minchromatin,maxchromatin);
					 normalisedSnucleoli=normalisedDouble(st.nextToken(),minnucleoli,maxnucleoli);
					 normalisedSmitosis=normalisedDouble(st.nextToken(),minmitosis,maxmitosis);
			}
		}


///MAPPERCLASS
				
		@Override
		// The map() method is run by MapReduce once for each row supplied as the input data
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException
		{
			// Tokenize the input line (presented as 'value' by MapReduce) from the csv file
			// This is the training dataset, R
			String rLine = value.toString();
			StringTokenizer st = new StringTokenizer(rLine, ",");
			double normalisedRcode=normalisedDouble(st.nextToken(),mincode,maxcode);
			double normalisedRclump=normalisedDouble(st.nextToken(),minclump,maxclump);
			double normalisedRcellsize=normalisedDouble(st.nextToken(),mincellsize,maxcellsize);
			double	normalisedRcellshape=normalisedDouble(st.nextToken(),mincellshape,maxcellshape);
			double normalisedRmarginal=normalisedDouble(st.nextToken(),minmarginal,maxmarginal);
			double normalisedRsinglecellsize=normalisedDouble(st.nextToken(),minsinglecellsize,maxsinglecellsize);
			double normalisedRbarenuclei=normalisedDouble(st.nextToken(),minbarenuclei,maxbarenuclei);
			double normalisedRchromatin=normalisedDouble(st.nextToken(),minchromatin,maxchromatin);
			double normalisedRnucleoli=normalisedDouble(st.nextToken(),minnucleoli,maxnucleoli);
			double normalisedRmitosis=normalisedDouble(st.nextToken(),minmitosis,maxmitosis);
			String type= st.nextToken();
			
			
			// Using these row specific values and the unchanging S dataset values, calculate a total squared
			// distance between each pair of corresponding values.
			double tDist = totalSquaredDistance(normalisedRcode, normalisedRclump, normalisedRcellsize, normalisedRcellshape,normalisedRmarginal,normalisedRsinglecellsize,normalisedRbarenuclei,normalisedRchromatin,normalisedRnucleoli,normalisedRmitosis,
					normalisedScode, normalisedSclump, normalisedScellsize, normalisedScellshape,normalisedSmarginal,normalisedSsinglecellsize,normalisedSbarenuclei,normalisedSchromatin,normalisedSnucleoli,normalisedSmitosis);		
			
			// Add the total distance and corresponding car model for this row into the TreeMap with distance
			// as key and model as value.
			KnnMap.put(tDist, type);
			// Only K distances are required, so if the TreeMap contains over K entries, remove the last one
			// which will be the highest distance number.
			if (KnnMap.size() > K)
			{
				KnnMap.remove(KnnMap.lastKey());
			}
		}

		@Override
		// The cleanup() method is run once after map() has run for every row
		protected void cleanup(Context context) throws IOException, InterruptedException
		{
			// Loop through the K key:values in the TreeMap
			for(Map.Entry<Double, String> entry : KnnMap.entrySet())
			{
				  Double knnDist = entry.getKey();
				  String knnModel = entry.getValue();
				  // distanceAndModel is the instance of DoubleString declared aerlier
				  distanceAndModel.set(knnDist, knnModel);
				  // Write to context a NullWritable as key and distanceAndModel as value
				  context.write(NullWritable.get(), distanceAndModel);
			}
		}
	}

	// The reducer class accepts the NullWritable and DoubleString objects just supplied to context and
	// outputs a NullWritable and a Text object for the final classification.
	public static class KnnReducer extends Reducer<NullWritable, DoubleString, NullWritable,Text>
	{
		TreeMap<Double, String> KnnMap = new TreeMap<Double, String>();
		int K=121;
		
		@Override
		// setup() again is run before the main reduce() method
		protected void setup(Context context) throws IOException, InterruptedException
		{
			if (context.getCacheFiles() != null && context.getCacheFiles().length > 0)
			{
				// Read parameter file using alias established in main()
				String knnParams = FileUtils.readFileToString(new File("test1.txt"));
				StringTokenizer st = new StringTokenizer(knnParams, ",");
				// Only K is needed from the parameter file by the reducer
				K = Integer.parseInt(st.nextToken());
			}
		}
		
		@Override
		// The reduce() method accepts the objects the mapper wrote to context: a NullWritable and a DoubleString
		public void reduce(NullWritable key, Iterable<DoubleString> values, Context context) throws IOException, InterruptedException
		{
			// values are the K DoubleString objects which the mapper wrote to context
			// Loop through these
			for (DoubleString val : values)
			{
				String rModel = val.getModel();
				double tDist = val.getDistance();
				//context.write(rModel,tDist);
				// Populate another TreeMap with the distance and model information extracted from the
				// DoubleString objects and trim it to size K as before.
				K=5;
				KnnMap.put(tDist, rModel);
				if (KnnMap.size() > K)
				{
					KnnMap.remove(KnnMap.lastKey());
				}
			}	

				// This section determines which of the K values (models) in the TreeMap occurs most frequently
				// by means of constructing an intermediate ArrayList and HashMap.

				// A List of all the values in the TreeMap.
				List<String> knnList = new ArrayList<String>(KnnMap.values());
				//context.write(NullWritable.get(),knnList.size());
				Map<String, Integer> freqMap = new HashMap<String, Integer>();
			    
			    // Add the members of the list to the HashMap as keys and the number of times each occurs
			    // (frequency) as values
			    for(int i=0; i< knnList.size(); i++)
			    {  
			        Integer frequency = freqMap.get(knnList.get(i));
			        if(frequency == null)
			        {
			            freqMap.put(knnList.get(i), 1);
			        } else
			        {
			            freqMap.put(knnList.get(i), frequency+1);
			        }
			    }
			    
			    // Examine the HashMap to determine which key (model) has the highest value (frequency)
			    String mostCommonModel = null;
			    int maxFrequency = -1;
			    for(Map.Entry<String, Integer> entry: freqMap.entrySet())
			    {
			        if(entry.getValue() > maxFrequency)
			        {
			            mostCommonModel = entry.getKey();
			            maxFrequency = entry.getValue();
			        }
			    }
			    
			// Finally write to context another NullWritable as key and the most common model just counted as value.
			context.write(NullWritable.get(), new Text(mostCommonModel)); // Use this line to produce a single classification
			//context.write(NullWritable.get(), new Text(KnnMap.toString()));	// Use this line to see all K nearest neighbours and distances
		}
	}

	// Main program to run: By calling MapReduce's 'job' API it configures and submits the MapReduce job.
	public static void main(String[] args) throws Exception
	{
		// Create configuration
		Configuration conf = new Configuration();
		
		if (args.length != 3)
		{
			System.err.println("Usage: KnnPattern <in> <out> <parameter file>");
			System.exit(2);
		}

		// Create job
		Job job = Job.getInstance(conf, "Find K-Nearest Neighbour");
		job.setJarByClass(KnnPattern.class);
		// Set the third parameter when running the job to be the parameter file and give it an alias
		job.addCacheFile(new URI(args[2] + "#knnParamFile")); // Parameter file containing test data
		
		// Setup MapReduce job
		job.setMapperClass(KnnMapper.class);
		job.setReducerClass(KnnReducer.class);
		job.setNumReduceTasks(1); // Only one reducer in this design

		// Specify key / value
		job.setMapOutputKeyClass(NullWritable.class);
		job.setMapOutputValueClass(DoubleString.class);
		job.setOutputKeyClass(NullWritable.class);
		job.setOutputValueClass(Text.class);
				
		// Input (the data file) and Output (the resulting classification)
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		
		// Execute job and return status
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}
