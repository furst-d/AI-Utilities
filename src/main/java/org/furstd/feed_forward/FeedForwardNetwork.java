package org.furstd.feed_forward;

import java.util.Scanner;

public class FeedForwardNetwork {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        FeedForwardNetwork ffnn = new FeedForwardNetwork();

        while (true) {
            System.out.println("\nMenu:");
            System.out.println("1) TODO");
            System.out.println("2) TODO");
            System.out.println("3) TODO");
            System.out.println("4) Exit");
            System.out.print("Input choice: ");

            int choice = scanner.nextInt();
            switch (choice) {
                case 1:
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
