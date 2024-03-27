package org.furstd.feed_forward;

import java.util.Scanner;

public class FeedForwardNetwork {

    public void computeResponse() {
        //TODO
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        FeedForwardNetwork ffnn = new FeedForwardNetwork();

        //TODO počet vrstev
        //TODO v cyklu načíst počet neuronů v každé vrstvě
        //TODO v cyklu načíst váhy mezi neurony

        while (true) {
            System.out.println("\nMenu:");
            System.out.println("1) Compute response");
            System.out.println("2) TODO");
            System.out.println("3) TODO");
            System.out.println("4) Exit");
            System.out.print("Input choice: ");

            int choice = scanner.nextInt();
            switch (choice) {
                case 1:
                    ffnn.computeResponse();
                    break;
                case 2:
                    break;
                case 3:
                    break;
                case 4:
                    System.out.println("Exiting...");
                    scanner.close();
                    return;
                default:
                    System.out.println("Invalid choice!");
                    break;
            }
        }
    }
}
