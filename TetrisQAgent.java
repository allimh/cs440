package src.pas.tetris.agents;


// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;


// JAVA PROJECT IMPORTS
import edu.bu.pas.tetris.agents.QAgent;
import edu.bu.pas.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.pas.tetris.game.Board;
import edu.bu.pas.tetris.game.Block;
import edu.bu.pas.tetris.game.Game.GameView;
import edu.bu.pas.tetris.game.minos.Mino;
import edu.bu.pas.tetris.linalg.Matrix;
import edu.bu.pas.tetris.nn.Model;
import edu.bu.pas.tetris.nn.LossFunction;
import edu.bu.pas.tetris.nn.Optimizer;
import edu.bu.pas.tetris.nn.models.Sequential;
import edu.bu.pas.tetris.nn.layers.Dense; // fully connected layer
import edu.bu.pas.tetris.nn.layers.ReLU;  // some activations (below too)
import edu.bu.pas.tetris.nn.layers.Tanh;
import edu.bu.pas.tetris.nn.layers.Sigmoid;
import edu.bu.pas.tetris.training.data.Dataset;
import edu.bu.pas.tetris.utils.Pair;
import edu.bu.pas.tetris.utils.Coordinate;


public class TetrisQAgent
    extends QAgent
{

    public static final double EXPLORATION_PROB = 0.05;

    private Random random;

    public TetrisQAgent(String name)
    {
        super(name);
        this.random = new Random(12345); // optional to have a seed
    }

    public Random getRandom() { return this.random; }

    @Override
    public Model initQFunction()
    {
        System.out.println("initQFunction called!");
        
        final int numPixelsInImage = Board.NUM_ROWS * Board.NUM_COLS;
        final int numExtraFeatures = 18; // based on the feature design (18 total)
        final int inputDim = numPixelsInImage + numExtraFeatures;

        final int hiddenDim1 = inputDim * 2; // first hidden layer wider
        final int hiddenDim2 = inputDim;     // second hidden layer smaller
        final int outDim = 1;                // predict a single Q-value

        Sequential qFunction = new Sequential();
        qFunction.add(new Dense(inputDim, hiddenDim1));
        qFunction.add(new ReLU());
        qFunction.add(new Dense(hiddenDim1, hiddenDim2));
        qFunction.add(new ReLU());
        qFunction.add(new Dense(hiddenDim2, outDim));

        return qFunction;
    }

    /**
        This function is for you to figure out what your features
        are. This should end up being a single row-vector, and the
        dimensions should be what your qfunction is expecting.
        One thing we can do is get the grayscale image
        where squares in the image are 0.0 if unoccupied, 0.5 if
        there is a "background" square (i.e. that square is occupied
        but it is not the current piece being placed), and 1.0 for
        any squares that the current piece is being considered for.
        
        We can then flatten this image to get a row-vector, but we
        can do more than this! Try to be creative: how can you measure the
        "state" of the game without relying on the pixels? If you were given
        a tetris game midway through play, what properties would you look for?
     */
    @Override
    public Matrix getQFunctionInput(final GameView game,
                                    final Mino potentialAction)
    {
        Matrix flattenedImage = null;
        try
        {
            flattenedImage = game.getGrayscaleImage(potentialAction).flatten();
            List<Double> extraFeatures = new ArrayList<>();

            // add and normalize added features
            Board board = game.getBoard();
            Block[][] grid = board.getBoard();

            int[] columnHeights = new int[Board.NUM_COLS];
            int maxHeight = 0;
            int totalHeight = 0;

            // total height and max height
            for (int x = 0; x < Board.NUM_COLS; x++) {
                for (int y = 0; y < Board.NUM_ROWS; y++) {
                    if (grid[y][x] != null) {
                        int height = Board.NUM_ROWS - y;
                        columnHeights[x] = height;
                        totalHeight += height;
                        maxHeight = Math.max(maxHeight, height);
                        break;
                    }
                }
            }

            extraFeatures.add(totalHeight / 200.0); // normalize
            extraFeatures.add(maxHeight / 20.0); // normalize

            // board smoothness (even/uneven col heights)
            int smoothness = 0;
            for (int i = 0; i < Board.NUM_COLS -1; i++) {
                smoothness += Math.abs(columnHeights[i] - columnHeights[i+1]);
            }

            extraFeatures.add(smoothness / 100.0); // normalize

            // total holes on the board
            int holes = 0;
            for (int x = 0; x < Board.NUM_COLS; x++) {
                boolean seenBlock = false;
                for (int y = 0; y < Board.NUM_ROWS; y++) {
                    if (grid[y][x] != null) {
                        seenBlock = true;
                    } else if (seenBlock) {
                        holes++;
                    }
                }
            }

            extraFeatures.add(holes / 100.0); // normalize

            // lines cleared
            extraFeatures.add(game.getScoreThisTurn() / 100.0); // normalize

            // orientation
            extraFeatures.addAll(findOrientation(potentialAction.getOrientation()));
            
            // mino type
            extraFeatures.addAll(minoType(potentialAction.getType()));

            // X/Y position
            Coordinate coord = potentialAction.getPivotBlockCoordinate();
            extraFeatures.add(coord.getXCoordinate() / (double) Board.NUM_COLS);
            extraFeatures.add(coord.getYCoordinate() / (double) Board.NUM_ROWS);

            // combine grayscale image + added features
            int totalLen = flattenedImage.numel() + extraFeatures.size();
            Matrix input = Matrix.zeros(1, totalLen);

            // Copy flattened grayscale
            for (int i = 0; i < flattenedImage.numel(); i++) {
                input.set(0, i, flattenedImage.get(0, i));
            }

            // Copy extra features
            for (int i = 0; i < extraFeatures.size(); i++) {
                input.set(0, flattenedImage.numel() + i, extraFeatures.get(i));
            }

            return input;

        } catch(Exception e)
        {
            e.printStackTrace();
            System.exit(-1);
        }
        return null;
    }

    // getQFunctionInput helper methods

    private List<Double> minoType(Mino.MinoType type) {
        List<Double> encoding = new ArrayList<>(Arrays.asList(0., 0., 0., 0., 0., 0., 0.));
        switch (type) {
            case I: encoding.set(0, 1.0); break;
            case O: encoding.set(1, 1.0); break;
            case T: encoding.set(2, 1.0); break;
            case S: encoding.set(3, 1.0); break;
            case Z: encoding.set(4, 1.0); break;
            case J: encoding.set(5, 1.0); break;
            case L: encoding.set(6, 1.0); break;
        }
        return encoding;
    }

    private List<Double> findOrientation(Mino.Orientation orientation) {
        List<Double> encoding = new ArrayList<>(Arrays.asList(0., 0., 0., 0.));
    
        switch (orientation) {
            case A: encoding.set(0, 1.0); break;
            case B: encoding.set(1, 1.0); break;
            case C: encoding.set(2, 1.0); break;
            case D: encoding.set(3, 1.0); break;
            default: break;
        }
    
        return encoding;
    }

    /**
     * This method is used to decide if we should follow our current policy
     * (i.e. our q-function), or if we should ignore it and take a random action
     * (i.e. explore).
     *
     * Remember, as the q-function learns, it will start to predict the same "good" actions
     * over and over again. This can prevent us from discovering new, potentially even
     * better states, which we want to do! So, sometimes we should ignore our policy
     * and explore to gain novel experiences.
     *
     * The current implementation chooses to ignore the current policy around 5% of the time.
     * While this strategy is easy to implement, it often doesn't perform well and is
     * really sensitive to the EXPLORATION_PROB. I would recommend devising your own
     * strategy here.
     */
    @Override
    public boolean shouldExplore(final GameView game,
                                 final GameCounter gameCounter)
    {
        // System.out.println("cycleIdx=" + gameCounter.getCurrentCycleIdx() + "\tgameIdx=" + gameCounter.getCurrentGameIdx());
        List<Mino> possibleActions = game.getFinalMinoPositions();
    double[] qValues = new double[possibleActions.size()];

    try {
        for (int i = 0; i < possibleActions.size(); i++) {
            Matrix input = getQFunctionInput(game, possibleActions.get(i));
            qValues[i] = this.getQFunction().forward(input).get(0,0);
        }

        // find max qValue
        double maxQ = Double.NEGATIVE_INFINITY;
        for (double q : qValues) {
            if (q > maxQ) maxQ = q;
        }

        // exp(q - maxQ) for numerical stability
        double qSum = 0.0;
        double[] probabilities = new double[qValues.length];
        for (int i = 0; i < qValues.length; i++) {
            probabilities[i] = Math.exp(qValues[i] - maxQ);
            qSum += probabilities[i];
        }
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= qSum;
        }

        // calculate entropy
        double entropy = 0.0;
        for (double p : probabilities) {
            if (p > 1e-8) entropy -= p * Math.log(p);
        }

        double entropyThreshold = 1.0 + Math.log1p(gameCounter.getCurrentGameIdx() / 5000.0);

        return entropy > entropyThreshold;

    } catch (Exception e) {
        e.printStackTrace();
        return this.getRandom().nextDouble() < EXPLORATION_PROB;
    }
    }

    /**
     * This method is a counterpart to the "shouldExplore" method. Whenever we decide
     * that we should ignore our policy, we now have to actually choose an action.
     *
     * You should come up with a way of choosing an action so that the model gets
     * to experience something new. The current implemention just chooses a random
     * option, which in practice doesn't work as well as a more guided strategy.
     * I would recommend devising your own strategy here.
     */

     /* Loops through all possible mino placements.
      * Scores them based on placement height, center distance, and rotation.
      * Picks the move with the highest exploration score.
      * If no clear winner (rare), randomly picks one.
      * Encourages low and safe placements, keeps the stack flexible early,
      * and promotes diverse plays late.
      */
    @Override
    public Mino getExplorationMove(final GameView game)
    {
        List<Mino> possibleActions = game.getFinalMinoPositions();
        Mino bestExplorationMove = null;
        double bestExplorationScore = Double.NEGATIVE_INFINITY;
    
        for (Mino candidate : possibleActions) {
            double explorationScore = evaluateMinoForExploration(candidate);
    
            if (explorationScore > bestExplorationScore) {
                bestExplorationScore = explorationScore;
                bestExplorationMove = candidate;
            }
        }
    
        // fallback to random if somehow no move found (very unlikely)
        return bestExplorationMove != null ? bestExplorationMove :
               possibleActions.get(this.getRandom().nextInt(possibleActions.size()));
    }

    private double evaluateMinoForExploration(Mino mino) {
        Coordinate pivot = mino.getPivotBlockCoordinate();
        int y = pivot.getYCoordinate();
        int x = pivot.getXCoordinate();
        double score = 0.0;

        // reward lower placements (higher Y = safer)
        score += y * 1.5; 

        // dynamic center placement preference 
        int center = Board.NUM_COLS / 2;
        double distanceFromCenter = Math.abs(x - center);

        double heightRatio = y / (double) Board.NUM_ROWS;

        double centerPenaltyWeight;
        if (heightRatio < 0.3) {
            centerPenaltyWeight = 1.0; // strong center preference (early game)
        } else if (heightRatio < 0.7) {
            centerPenaltyWeight = 0.5; // moderate center preference (mid game)
        } else {
            centerPenaltyWeight = 0.2; // weak center preference (high stack)
        }

        score -= distanceFromCenter * centerPenaltyWeight;

        // minor bonus for different orientations (encourages rotation exploration)
        switch (mino.getOrientation()) {
            case A: score += 0.1; break;
            case B: score += 0.2; break;
            case C: score += 0.15; break;
            case D: score += 0.05; break;
        }

        return score;
    }

    /**
     * This method is called by the TrainerAgent after we have played enough training games.
     * In between the training section and the evaluation section of a cycle, we need to use
     * the exprience we've collected (from the training games) to improve the q-function.
     *
     * You don't really need to change this method unless you want to. All that happens
     * is that we will use the experiences currently stored in the replay buffer to update
     * our model. Updates (i.e. gradient descent updates) will be applied per minibatch
     * (i.e. a subset of the entire dataset) rather than in a vanilla gradient descent manner
     * (i.e. all at once)...this often works better and is an active area of research.
     *
     * Each pass through the data is called an epoch, and we will perform "numUpdates" amount
     * of epochs in between the training and eval sections of each cycle.
     */
    @Override
    public void trainQFunction(Dataset dataset,
                               LossFunction lossFunction,
                               Optimizer optimizer,
                               long numUpdates)
    {
        for(int epochIdx = 0; epochIdx < numUpdates; ++epochIdx)
        {
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix> > batchIterator = dataset.iterator();

            while(batchIterator.hasNext())
            {
                Pair<Matrix, Matrix> batch = batchIterator.next();

                try
                {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());

                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(),
                                                  lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch(Exception e)
                {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }

    /**
     * This method is where you will devise your own reward signal. Remember, the larger
     * the number, the more "pleasurable" it is to the model, and the smaller the number,
     * the more "painful" to the model.
     *
     * This is where you get to tell the model how "good" or "bad" the game is.
     * Since you earn points in this game, the reward should probably be influenced by the
     * points, however this is not all. In fact, just using the points earned this turn
     * is a **terrible** reward function, because earning points is hard!!
     *
     * I would recommend you to consider other ways of measuring "good"ness and "bad"ness
     * of the game. For instance, the higher the stack of minos gets....generally the worse
     * (unless you have a long hole waiting for an I-block). When you design a reward
     * signal that is less sparse, you should see your model optimize this reward over time.
     */
    @Override
    public double getReward(final GameView game)
    {
        double reward = 0.0;

        Board board = game.getBoard();
        Block[][] grid = board.getBoard();

        // set values for num holes, max height, and total height
        int holes = 0;
        int maxHeight = 0;
        int totalHeight = 0;
        for (int x = 0; x < Board.NUM_COLS; x++) {
            boolean seenBlock = false;
            for (int y = 0; y < Board.NUM_ROWS; y++) {
                if (grid[y][x] != null) {
                    if (!seenBlock) {
                        int height = Board.NUM_ROWS - y;
                        totalHeight += height;
                        maxHeight = Math.max(maxHeight, height);
                        seenBlock = true;
                    }
                } else if (seenBlock) {
                    holes++;
                }
            }
        } 

        

        // num lines cleared
        int linesCleared = countClearedLines(board);

        double heightRatio = maxHeight / (double) Board.NUM_ROWS;
        double panicScale = Math.min(2.0, 1.0 + heightRatio * 3.0); // cap the panic scale slightly
        
        if (linesCleared == 4) {
            reward += 10.0 * panicScale;
        } else {
            reward += linesCleared * 5.0 * panicScale;
        }

        reward -= Math.min(holes * 2.0 * panicScale, 50.0);  // cap hole penalty at -50
        reward -= Math.max(0, maxHeight - 10) * 0.5 * panicScale;
        reward -= Math.min(totalHeight / 50.0 * panicScale, 30.0); // cap total height penalty at -30

        int smoothness = getSmoothness(grid);
        reward -= Math.min(smoothness / 100.0 * panicScale, 20.0); // cap roughness penalty at -20

        reward += game.getScoreThisTurn() * 0.1 * panicScale;

        // small survival bonus
        reward += 0.5;

        return reward;
    }

    private int countClearedLines(Board board) {
        int cleared = 0;
        Block[][] grid = board.getBoard();
        for (int y = 0; y < Board.NUM_ROWS; y++) {
            boolean fullLine = true;
            for (int x = 0; x < Board.NUM_COLS; x++) {
                if (grid[y][x] == null) {
                    fullLine = false;
                    break;
                }
            }
            if (fullLine) cleared++;
        }
        return cleared;
    }


    private int getSmoothness(Block[][] grid) {
        int[] columnHeights = new int[Board.NUM_COLS];
        for (int x = 0; x < Board.NUM_COLS; x++) {
            for (int y = 0; y < Board.NUM_ROWS; y++) {
                if (grid[y][x] != null) {
                    columnHeights[x] = Board.NUM_ROWS - y;
                    break;
                }
            }
        }
        int smoothness = 0;
        for (int i = 0; i < Board.NUM_COLS - 1; i++) {
            smoothness += Math.abs(columnHeights[i] - columnHeights[i + 1]);
        }
        return smoothness;
    }

    

}
