package algo;

import org.opencv.core.Core;

import javafx.application.Application;
import javafx.stage.Stage;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;

/**
 * The main class for a JavaFX application. It creates and handle the main
 * window with its resources (style, graphics, etc.).
 * 
 * This application handles a video stream and try to find any possible human
 * face in a frame. It can use the Haar or the LBP classifier.
 * 
 * @author Luigi De Russis / Igor @ HeroinSoul / Jim O'Connorhorrill @ cuerobotics
 * @version 1.3 (2016-08-04)
 * @since 1.0 (2014-01-10)
 * 
 */
public class Main extends Application
{
	@Override
	public void start(Stage primaryStage)
	{
		try
		{
                        // crea la escena con la interfaz de usuario
			FXMLLoader loader = new FXMLLoader(getClass().getResource("JFXoverlay.fxml"));
			BorderPane root = (BorderPane) loader.load();
			
			root.setStyle("-fx-background-color: whitesmoke;");
			
			Scene scene = new Scene(root, 800, 600);
			scene.getStylesheets().add(getClass().getResource("application.css").toExternalForm());
			
			primaryStage.setTitle("Face Detection and Tracking");
			primaryStage.setScene(scene);
			
			primaryStage.show();
			
			// inicializa los controladores
			FXController controller = loader.getController();
			controller.init();
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args)
	{
		// carga la libreria nativa de OpenCV
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		launch(args);
	}
}
