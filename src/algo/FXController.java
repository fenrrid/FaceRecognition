package algo;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FilenameFilter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.face.Face;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.video.Video;
import org.opencv.videoio.VideoCapture;


import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import org.opencv.face.BasicFaceRecognizer;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.face.FaceRecognizer;





public class FXController
{
	///// Declaraciones FXML
	@FXML
	private Button cameraButton;
	@FXML
	private ImageView originalFrame;
	@FXML
	private CheckBox haarClassifier;
	@FXML
	private CheckBox lbpClassifier;
	@FXML
	private CheckBox newUser;
	@FXML
	private TextField newUserName;
	@FXML
	private Button newUserNameSubmit;
	/////////////////////////////////
	
	///////Declaraciones para el funcionamiento con OpenCV
	private ScheduledExecutorService timer;
	// Video captura de openCv
	private VideoCapture capture;
	private boolean cameraActive;
	// Clasificador de cara 
	private CascadeClassifier faceCascade;
	private int absoluteFaceSize;
	public int index = 0;
	public int ind = 0;
	// Nombre para usuarios nuevos
	public String newname;
	// Nombres de las personas agregadas al set de entrenamiento
	public HashMap<Integer, String> names = new HashMap<Integer, String>();
	public int random = (int )(Math.random() * 20 + 3);
	
        //////////inicializa los controladores para el funcionamiento
	public void init()
	{
                //Inicializa la videoCaptura
		this.capture = new VideoCapture();
                //Inicializa el clasificador de cara (frontal)
		this.faceCascade = new CascadeClassifier("C:\\Users\\danie\\Desktop\\algo\\src\\resources\\haarcascades\\haarcascade_frontalface_alt.xml");
		this.absoluteFaceSize = 0;
		// Desabilita la opcion "nuevo usuario"
		this.newUserNameSubmit.setDisable(true);
		this.newUserName.setDisable(true);
		// entrenamiento (opcional) este paso puede omitirse, de esta forma el programa no entrena cada vez que inicia
		trainModel();
	}
	


	//Inicializacion de camara
	@FXML
	protected void startCamera()
	{
		// aigna un tamaño al frame
		originalFrame.setFitWidth(600);
		originalFrame.setPreserveRatio(true);
		
		if (!this.cameraActive)
		{
			// disable setting checkboxes
			this.haarClassifier.setDisable(true);
			this.lbpClassifier.setDisable(true);
			

			this.newUser.setDisable(true);
			
			// inicia la videocaptura del video
			this.capture.open(0);
			

			if (this.capture.isOpened())
			{
				this.cameraActive = true;
				
				// toma un frame cada 33 ms (30 frames/sec)
				Runnable frameGrabber = new Runnable() {
					
					@Override
					public void run()
					{
						Image imageToShow = grabFrame();
						originalFrame.setImage(imageToShow);
					}
				};
				
				this.timer = Executors.newSingleThreadScheduledExecutor();
				this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);
				this.cameraButton.setText("Stop Camera");
			}
			else
			{
				// log de error
				System.err.println("Failed to open the camera connection...");
			}
		}
		else
		{
                        //////////Al desactivarse la camara
			this.cameraActive = false;
			this.cameraButton.setText("Start Camera");
			// //////////////////////////////////
			this.haarClassifier.setDisable(false);
			this.lbpClassifier.setDisable(false);
			// activa el boton nuevo usuario
			this.newUser.setDisable(false);
			
			try
			{
				this.timer.shutdown();
				this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
			}
			catch (InterruptedException e)
			{
				// log the exception
				System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
			}
			
			// libera la camara
			this.capture.release();
			// limpia el frame
			this.originalFrame.setImage(null);
			
			// limpia los parametros
			index = 0;
			newname = "";
		}
	}
	
	//////Obtiene un frame del video, si selecciona alguna regresa una imagen
	private Image grabFrame()
	{
		// Inicializa todo
		Image imageToShow = null;
		Mat frame = new Mat();
		
		// Verifica si la captura esta activa
		if (this.capture.isOpened())
		{
			try
			{
				// Lee el frame actual
				this.capture.read(frame);
				
				// Si no esta vacio lo lee
				if (!frame.empty())
				{
					// Detecta el rostro
					this.detectAndDisplay(frame);
					
					// Combierte el objeto en opencv a javaFX
					imageToShow = mat2Image(frame);
				}
				
			}
			catch (Exception e)
			{
				// log el error completo
				System.err.println("ERROR: " + e);
			}
		}
		
		return imageToShow;
	}
	
	/////////Entrenador
	private void trainModel () {
		// Lee los datos del set de entrenamiento
				File root = new File("C:\\Users\\danie\\Desktop\\algo\\src\\resources\\trainingset\\combined");
				FilenameFilter imgFilter = new FilenameFilter() {
		            public boolean accept(File dir, String name) {
		                name = name.toLowerCase();
		                return name.endsWith(".png");
		            }
		        };
		        
		        File[] imageFiles = root.listFiles(imgFilter);
		        List<Mat> images = new ArrayList<Mat>();
		        System.out.println("Imagenes leidas: " + imageFiles.length);
		        List<Integer> trainingLabels = new ArrayList<>();
		        Mat labels = new Mat(imageFiles.length,1,CvType.CV_32SC1);
		        int counter = 0;
		        
		        for (File image : imageFiles) {
		        	// convierte las imagenes a un tipo de vairable de OpenCV
		        	Mat img = Imgcodecs.imread(image.getAbsolutePath());
		        	// cambia la imagen a escala de grises y ecualiza el histograma
		        	Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2GRAY);
		        	Imgproc.equalizeHist(img, img);
		        	// Extrae el nombre del archivo 
		        	int label = Integer.parseInt(image.getName().split("\\-")[0]);
		        	// Extrae el nombre de la persona del archivo y lo añade a los hashmaps
		        	String labnname = image.getName().split("\\_")[0];
		        	String name = labnname.split("\\-")[1];
		        	names.put(label, name);
		        	images.add(img);
		        	labels.put(counter, 0, label);
		        	counter++;
		        }
                                //Reconoce el rostro
                                FaceRecognizer faceRecognizer = Face.createLBPHFaceRecognizer();
                                //Entrena la red con el set de imagenes procesado
		        	faceRecognizer.train(images, labels);
                                //Guarda el entrenamiento en un archivo
		        	faceRecognizer.save("traineddata");
	}
	
	// Metodo de reconocimiento de rosto
        //Identifica el rostro, busca coincidencias y pone un nombre en el rostro identificado
        //recibe el rostro identificado
	private double[] faceRecognition(Mat currentFace) {
       	
        	// Label de prediccion
        	
        	int[] predLabel = new int[1];
            double[] confidence = new double[1];
            int result = -1;
            
            FaceRecognizer faceRecognizer = Face.createLBPHFaceRecognizer();
            //llama los datos del entrenamiento
            faceRecognizer.load("traineddata");
                //Manda a llamar el metodo de prediccion 
        	faceRecognizer.predict(currentFace,predLabel,confidence);
        	result = predLabel[0];
        	//regresa el resultado de la prediccion y el valor de la coincidencia
        	return new double[] {result,confidence[0]};
	}
	
	
	//Metodo de deteccion de rostro y tranking.
	private void detectAndDisplay(Mat frame)
	{
		MatOfRect faces = new MatOfRect();
		Mat grayFrame = new Mat();
		
		// Convierte el frame a escala de grises
		Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
		// Ecualiza el histograma
		Imgproc.equalizeHist(grayFrame, grayFrame);
		
		// Computa el tamaño del rostro minimo
		if (this.absoluteFaceSize == 0)
		{
			int height = grayFrame.rows();
			if (Math.round(height * 0.2f) > 0)
			{
				this.absoluteFaceSize = Math.round(height * 0.2f);
			}
		}
		
		// Detecta rostros
		this.faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE,
				new Size(this.absoluteFaceSize, this.absoluteFaceSize), new Size());
				
		// Dibuja rectangulos en los rostros
		Rect[] facesArray = faces.toArray(); 
		for (int i = 0; i < facesArray.length; i++) {
			Imgproc.rectangle(frame, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0), 3);

			// Corta los rostros detectados
			Rect rectCrop = new Rect(facesArray[i].tl(), facesArray[i].br());
			Mat croppedImage = new Mat(frame, rectCrop);
			// Los cambia a escala de grises
			Imgproc.cvtColor(croppedImage, croppedImage, Imgproc.COLOR_BGR2GRAY);
			// Ecualiza el histograma
			Imgproc.equalizeHist(croppedImage, croppedImage);
			// Cambia el tamaño de la imagen a un tamaño por default
			Mat resizeImage = new Mat();
			Size size = new Size(250,250);
			Imgproc.resize(croppedImage, resizeImage, size);
			
			// Verifica si el boton "nuevo usuario" esta seleccionado, si lo esta recolecta informacion
			if ((newUser.isSelected() && !newname.isEmpty())) {
				if (index<5) {
					Imgcodecs.imwrite("C:\\Users\\danie\\Desktop\\algo\\src\\resources\\trainingset\\combined\\" +
					random + "-" + newname + "_" + (index++) + ".png", resizeImage);
				}
			}
                        //llamado del metodo para reconocer el rostro (recive imagen actual) 
			double[] returnedResults = faceRecognition(resizeImage);
                        //Asigna variables para la prediccion del rostro
			double prediction = returnedResults[0];
			double confidence = returnedResults[1];
			

			int label = (int) prediction;
                        
			String name = "";
                        //si la el nombre es encontrado y el pocentaje de coincidencia es menor a 20(editable)
                        //el nombre es puesto en el cuadro, de lo contrario el nombre es "Desconocido"
			if (names.containsKey(label) && confidence<20) {
				name = names.get(label);
			} else {
				name = "Unknown";
			}
			
			//Crea el texto que se pondra en el recuadro, contiene el nombre y el porcentaje de coincidencia
                        // nota* entre menos sea el valor de coincidencia mayor es la coincidencia encontrada.
                        // este valor puede omitirse para que solo se muestre el nombre.
            String box_text = "Usuario = " + name + confidence;
            // Calcula la posicion del texto
            double pos_x = Math.max(facesArray[i].tl().x - 10, 0);
            double pos_y = Math.max(facesArray[i].tl().y - 10, 0);
            // agrega el texto al frame
            Imgproc.putText(frame, box_text, new Point(pos_x, pos_y), 
            		Core.FONT_HERSHEY_PLAIN, 1.0, new Scalar(0, 255, 0, 2.0));
		}
	}

	
        // Metodo de nuevo usuario
	@FXML
	protected void newUserNameSubmitted() {
		if ((newUserName.getText() != null && !newUserName.getText().isEmpty())) {
			newname = newUserName.getText();
			System.out.println("¡Usuario agregado!");
			newUserName.clear();
		}
	}
	

	/**
	 * The action triggered by selecting the Haar Classifier checkbox. It loads
	 * the trained set to be used for frontal face detection.
	 */
	@FXML
	protected void haarSelected(Event event)
	{
		// check whether the lpb checkbox is selected and deselect it
		if (this.lbpClassifier.isSelected())
			this.lbpClassifier.setSelected(false);
			
		this.checkboxSelection("C:\\Users\\danie\\Desktop\\algo\\src\\resources\\haarcascades\\haarcascade_frontalface_alt.xml");
	}
	
	/**
	 * The action triggered by selecting the LBP Classifier checkbox. It loads
	 * the trained set to be used for frontal face detection.
	 */
	@FXML
	protected void lbpSelected(Event event)
	{
		// check whether the haar checkbox is selected and deselect it
		if (this.haarClassifier.isSelected())
			this.haarClassifier.setSelected(false);
			
		this.checkboxSelection("C:\\Users\\danie\\Desktop\\algo\\src\\resources\\haarcascades\\haarcascade_frontalcatface.xml");
	}
	
        
	@FXML
	protected void newUserSelected(Event event) {
		if (this.newUser.isSelected()){
			this.newUserNameSubmit.setDisable(false);
			this.newUserName.setDisable(false);
		} else {
			this.newUserNameSubmit.setDisable(true);
			this.newUserName.setDisable(true);
		}
	}
	
	/**
	 * Method for loading a classifier trained set from disk
	 * 
	 * @param classifierPath
	 *            the path on disk where a classifier trained set is located
	 */
	private void checkboxSelection(String classifierPath)
	{
		// load the classifier(s)
		this.faceCascade.load(classifierPath);
		
		// now the video capture can start
		this.cameraButton.setDisable(false);
	}
	
	
	 // Convierte un object Mat (OpenCV) en la correspondiente Imagen para JavaFX
	private Image mat2Image(Mat frame)
	{
		// Crea un buffer temporal
		MatOfByte buffer = new MatOfByte();
		// Codifica el frame en el buffer acorde al formato PNG
		Imgcodecs.imencode(".png", frame, buffer);
		// Crea y regresa la imagen.
		return new Image(new ByteArrayInputStream(buffer.toArray()));
	}
	
}
